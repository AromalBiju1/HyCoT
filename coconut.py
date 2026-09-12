# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
MAX_N_LATENT = 8


def _patch_linear_attention_cache_for_training():
    """
    Monkey-patch transformers' LinearAttentionLayer.update_recurrent_state /
    update_conv_state to reassign the state tensor out-of-place instead of
    mutating it in place with .copy_().

    Root cause (confirmed against the installed transformers source,
    cache_utils.py): both methods intentionally use `.copy_()` into a
    persistent buffer so the tensor's memory address stays stable across
    calls -- this is a CUDA-graph capture/replay optimization for inference.
    But it means: within a single forward call, the layer reads the buffer
    as `initial_state`, fla's ChunkGatedDeltaRuleFunction.forward saves that
    exact tensor object for its own backward (see fla/ops/gated_delta_rule/
    chunk.py, ctx.save_for_backward(..., initial_state, ...)), and then
    update_recurrent_state's .copy_() overwrites that same buffer's contents
    in place -- before the forward call even returns. Autograd then fails at
    .backward() with "modified by an inplace operation", because the tensor
    it needs to read no longer holds the value it had when it was saved.

    This is not specific to Coconut's multi-pass reuse -- it would break
    ANY use of use_cache=True combined with .backward() on this hybrid
    architecture, since the static-address optimization was written for
    inference and is fundamentally incompatible with training through the
    cache.

    Fix: reassign the list entry to a new tensor instead of copying into
    the old one. This costs one extra small tensor allocation per layer per
    step (state shape is tiny -- e.g. [1, num_heads, head_dim, head_dim])
    and gives up CUDA-graph static addressing, which Coconut's training loop
    doesn't use anyway. Call this once, before training starts.
    """
    from transformers import cache_utils

    if not hasattr(cache_utils, "LinearAttentionLayer"):
        raise AttributeError(
            "transformers.cache_utils.LinearAttentionLayer not found -- this patch "
            "was written against transformers==5.17.0's cache_utils.py. Your installed "
            "version may have renamed/restructured this class. Run "
            "`print([n for n in dir(cache_utils) if 'Linear' in n or 'Layer' in n])` "
            "to find the current name and update this patch accordingly -- do not "
            "silently skip this, since without it training will hit the in-place "
            "gradient error again."
        )

    def _patched_update_recurrent_state(self, recurrent_states, state_idx=0, **kwargs):
        if not self.is_recurrent_states_initialized[state_idx]:
            self.lazy_initialization(recurrent_states=recurrent_states, state_idx=state_idx)
        # Out-of-place: preserves autograd's saved-for-backward reference to
        # this buffer's prior contents instead of overwriting them in place.
        self.recurrent_states[state_idx] = recurrent_states
        return self.recurrent_states[state_idx]

    def _patched_update_conv_state(
        self, conv_states, state_idx=0, conv_kernel_size=None, **kwargs
    ):
        if not self.is_conv_states_initialized[state_idx]:
            self.lazy_initialization(
                conv_states=conv_states, state_idx=state_idx, conv_kernel_size=conv_kernel_size
            )

        if not self.has_previous_state[state_idx]:
            full_conv_states = conv_states
            self.has_previous_state[state_idx] = True
            if (
                not self.record_past
                and full_conv_states.shape[-1] < self.conv_kernel_size[state_idx]
            ):
                padding_length = self.conv_kernel_size[state_idx] - full_conv_states.shape[-1]
                full_conv_states = torch.nn.functional.pad(
                    full_conv_states, (padding_length, 0), value=0
                )
        else:
            full_conv_states = torch.cat([self.conv_states[state_idx], conv_states], dim=-1)

        if not self.record_past:
            # Out-of-place (was: self.conv_states[state_idx].copy_(...))
            self.conv_states[state_idx] = full_conv_states[..., -self.conv_kernel_size[state_idx] :]
        else:
            self.conv_states[state_idx] = full_conv_states

        return full_conv_states

    cache_utils.LinearAttentionLayer.update_recurrent_state = _patched_update_recurrent_state
    cache_utils.LinearAttentionLayer.update_conv_state = _patched_update_conv_state


# Applied at import time so `import coconut` is enough -- no separate call needed.
_patch_linear_attention_cache_for_training()


def _clone_cache_states(kv_cache):
    """
    Clone the mutable recurrent/conv state tensors inside a hybrid
    DeltaNet+attention cache (e.g. Qwen3.5's DynamicCache of per-layer
    LinearAttentionLayer / attention-layer objects) before handing the cache
    to the next forward pass.

    Why this exists: HF's reference `torch_chunk_gated_delta_rule` kernel
    updates `recurrent_states` (and `causal_conv1d`-path updates
    `conv_states`) IN PLACE during each forward call. Coconut re-enters the
    model multiple times, passing the same cache object forward each time
    (`past_key_values=kv_cache`). Without cloning, pass N+1 mutates the exact
    tensor pass N's backward graph still references, so autograd raises
    "variable needed for gradient computation has been modified by an
    inplace operation" on `.backward()`.

    Cloning (never detaching) breaks the aliasing while keeping the state
    fully differentiable -- gradients still flow back through the cloned
    tensor to whatever produced it, which is required for continuous-thought
    training to actually train anything upstream of the state update.

    This function is defensive about cache internals: it walks whatever
    layer/attribute structure is present and clones any tensor it finds
    under attribute names containing "state" (conv_states, recurrent_states,
    or any future-renamed equivalent), and dict-wrapped tensors within them.
    Attention-only layers (standard KV cache) are left untouched -- ordinary
    KV tensors aren't mutated in place by attention and don't need this.
    """
    if kv_cache is None:
        return kv_cache

    # Try the common "cache has a list of per-layer objects" shape first
    # (DynamicCache-style). Fall back to introspecting the cache object
    # itself if that attribute doesn't exist.
    layers = getattr(kv_cache, "layers", None)
    if layers is None:
        # Older/alternate cache shapes sometimes store layers elsewhere.
        layers = getattr(kv_cache, "key_cache", None) or []

    for layer in layers:
        for attr_name in list(vars(layer).keys()) if hasattr(layer, "__dict__") else []:
            if "state" not in attr_name:
                continue
            value = getattr(layer, attr_name)
            if torch.is_tensor(value):
                setattr(layer, attr_name, value.clone())
            elif isinstance(value, dict):
                cloned = {
                    k: (v.clone() if torch.is_tensor(v) else v)
                    for k, v in value.items()
                }
                setattr(layer, attr_name, cloned)
            # else: leave non-tensor, non-dict attributes untouched

    return kv_cache


class Coconut(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
    ):

        super(Coconut, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id

        # tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):

        assert input_ids.shape[0] == 1, (
            "Qwen3.5 patch only supports batch_size=1 -- the original truncation "
            "logic assumed multi-instance batches with misaligned latent counts; "
            "at batch_size=1 it was a no-op, so it's removed rather than made "
            "DeltaNet-compatible. Use gradient_accumulation_steps for throughput."
        )

        logits = []

        latent_indices = (
            input_ids == self.latent_token_id
        ).nonzero()  # (num_latent_tokens_in_the_batch, 2)

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]  # bs, num_latent_tokens_in_the_instance (difference across the batch)

        max_n_latents = max([len(l) for l in latent_lists])

        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())
            # before the earliest latent token position

        kv_cache = None

        for pass_idx in range(max_n_latents):

            if kv_cache == None:
                # first forward pass
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0

            else:
                # Qwen3.5 patch: at batch_size=1 this truncation was always a
                # no-op (next_compute_range[0] == the cache's current length),
                # so pass kv_cache straight through instead of slicing it.
                # DeltaNet's conv_states/recurrent_states can't be sliced like
                # attention KV anyway -- see project notes.
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    past_key_values=kv_cache,
                    output_hidden_states=True,
                )

                hidden_states_offset = next_compute_range[0]
                # when we use kv_cache for the first k tokens
                # in `outputs.hidden_states`, [0, k) will be skipped
                # so we need to keep this offset to correctly use the last hidden states

            logits.append(outputs.logits)

            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            hidden_states = outputs.hidden_states[
                -1
            ]  # Get the last layer hidden states

            # Clone DeltaNet recurrent/conv states before this cache gets fed
            # into the next pass -- see _clone_cache_states docstring. This is
            # the fix for the "modified by an inplace operation" backward
            # error: each pass now gets its own tensor version, so autograd's
            # saved-for-backward references from pass_idx stay valid even
            # after pass_idx+1 runs.
            kv_cache = _clone_cache_states(outputs.past_key_values)

            # feedback the continuous thoughts to the input_embeds

            # first decide the positions to feedback
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            # to avoid in-place operations
            # break down inputs_embeds (bs, len, hidden_size) into a list of list of 1-d tensors
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            # replace some of them with continuous thoughts
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair

                # replace it with the preceding last hidden states
                tensor_list[batch_idx][token_idx] = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]

            # assemble the new inputs_embeds
            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )

        # final pass
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0] : next_compute_range[1], :
            ],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=kv_cache,  # Qwen3.5 patch: same no-op removal as above, now cloned per pass
            output_hidden_states=True,
        )

        logits.append(outputs.logits)

        self.gen_forward_cnt += max_n_latents + 1

        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    def train(self):
        self.base_causallm.train()

    def eval(self):
        self.base_causallm.eval()

    def generate(
        self,
        input_ids,
        attention_mask,  # attention_mask is not used
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        **kwargs
    ):

        self.gen_forward_cnt = 0

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        tokens = input_ids[0].detach().tolist()

        labels = input_ids.clone()  # placeholder. not used.
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),
        )
        inputs_embeds = outputs.inputs_embeds

        # get the first token using the current hidden state
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        new_token_embed = self.embedding(
            torch.tensor(next_token, device=input_ids.device)
        ).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        # get other tokens
        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            new_token_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        if synced_gpus:
            # in FSDP, the number of forward pass need to be the same across devices
            while (
                self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT
            ):  # leave some room for latent tokens
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)

        if output_embedding:
            # for analysis purpose
            return torch.tensor(tokens).view(1, -1), new_inputs_embeds

        else:
            return torch.tensor(tokens).view(1, -1)
