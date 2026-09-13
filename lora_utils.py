# LoRA helpers shared by run_single_gpu.py and smoke_test_coconut.py.
#
# Why a shared module: both scripts need the same target-module detection,
# LoRA-wrap logic, and trainable-only state-dict save. Duplicating this in
# both files would drift the moment one gets edited and not the other.

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType


def get_lora_target_modules(model):
    """
    Auto-detect Linear-layer leaf names to target for LoRA.

    Why auto-detect instead of hardcoding a name list: Qwen3.5's hybrid
    DeltaNet+attention backbone has two different layer families
    (linear_attn / full attention) whose internal Linear submodule names
    have never been verified against any published LoRA target_modules list
    -- those lists (q_proj/k_proj/v_proj/o_proj etc.) were written for plain
    Llama/GPT2-style attention. Walking the real model and collecting every
    nn.Linear leaf name is the only way to be sure DeltaNet's projections
    (whatever they're actually called -- in_proj/out_proj/beta_proj/etc.)
    get included, not silently skipped.

    Excludes lm_head (output projection, not something LoRA should touch
    here) and Embedding layers (embed_tokens is nn.Embedding, never matches
    isinstance Linear anyway, but noted for clarity).
    """
    target_suffixes = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.split(".")[-1]
        if leaf == "lm_head":
            continue
        target_suffixes.add(leaf)
    return sorted(target_suffixes)


def apply_lora(model, configs):
    """
    Wrap `model` (the raw AutoModelForCausalLM, already resized for the new
    latent/start/end tokens, NOT yet wrapped in Coconut) with a LoRA adapter.

    Freezes the entire backbone; only LoRA A/B matrices get requires_grad=True.
    Call this BEFORE wrapping in Coconut -- Coconut just proxies forward()
    calls through to whatever base_causallm it's given, so a PeftModel works
    identically to a raw AutoModelForCausalLM from Coconut's point of view.

    Config knobs (all optional, sane defaults if the yaml doesn't set them):
      lora_r              rank, default 16
      lora_alpha          scaling, default 32
      lora_dropout        default 0.05
      lora_target_modules comma-separated override string, or "auto"/unset
                          to use get_lora_target_modules() above
      lora_modules_to_save comma-separated module names to leave FULLY
                          trainable (not LoRA-adapted, just unfrozen) on top
                          of the LoRA adapters. Use this for embed_tokens/
                          lm_head if the new latent/start/end token rows
                          need to actually learn -- get_peft_model() freezes
                          everything not in target_modules OR modules_to_save
                          by default, and Embedding layers never match
                          target_modules (only nn.Linear leaves do), so
                          without this the new token rows stay frozen at
                          whatever <<'s embedding was copied to them.
    """
    target_modules = getattr(configs, "lora_target_modules", None)
    if target_modules in (None, "auto", "None", ""):
        target_modules = get_lora_target_modules(model)
        print(f"LoRA target_modules (auto-detected from model): {target_modules}")
    else:
        target_modules = [t.strip() for t in target_modules.split(",") if t.strip()]
        print(f"LoRA target_modules (from config override): {target_modules}")

    modules_to_save = getattr(configs, "lora_modules_to_save", None)
    if modules_to_save:
        modules_to_save = [m.strip() for m in modules_to_save.split(",") if m.strip()]
        print(f"Modules kept fully trainable (not LoRA, just unfrozen): {modules_to_save}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=getattr(configs, "lora_r", 16),
        lora_alpha=getattr(configs, "lora_alpha", 32),
        lora_dropout=getattr(configs, "lora_dropout", 0.05),
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def get_trainable_state_dict(model):
    """
    Return only the parameters that actually require grad (LoRA A/B
    matrices, plus later the MLP bridge if it's a submodule of `model`).

    Why this exists: torch.save(model.state_dict()) saves EVERY parameter
    regardless of requires_grad -- including the full frozen 4B backbone.
    That defeats the entire point of switching to LoRA (checkpoint files
    would still be full-model-sized). This filters state_dict() down to
    only the keys whose corresponding parameter has requires_grad=True,
    so checkpoints become megabytes instead of gigabytes.

    Loaded back with model.load_state_dict(saved, strict=False) -- strict
    must stay False since the frozen backbone keys are intentionally absent
    (they come from the base model download, not the checkpoint).
    """
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v for k, v in model.state_dict().items() if k in trainable_names}


def _unwrap_to_base(model):
    """
    PeftModel wraps the real HF model (LoraModel -> the actual
    Qwen3_5ForCausalLM). get_input_embeddings()/set_input_embeddings() are
    proxied by PeftModel, but there's no generic setter for .lm_head, so
    anything touching .lm_head directly needs the real underlying model,
    not the Peft wrapper. get_base_model() is Peft's own API for this and
    exists on any PeftModel; a raw (non-LoRA) model just returns itself.
    """
    return model.get_base_model() if hasattr(model, "get_base_model") else model


class SparseTrainableEmbeddingPatch(nn.Module):
    """
    Wraps a frozen nn.Embedding so a handful of specific token ids (here:
    Coconut's <|start-latent|>/<|latent|>/<|end-latent|>) get their own
    small trainable embedding table, instead of requiring the ENTIRE
    embedding matrix to be unfrozen just so 3 rows can move.

    This replaces the earlier `lora_modules_to_save: embed_tokens,lm_head`
    approach, which was the actual OOM cause: PEFT's modules_to_save
    unfreezes the whole module it's given. For this model that's
    248,080 x 2560 = ~635M params PER tensor (embed_tokens AND lm_head,
    since PEFT keeps separate trainable copies even though they're tied) --
    ~1.27B extra trainable params, ~10GB of AdamW optimizer state alone on
    top of everything else already on the T4. The actual task only needs 3
    rows to move. This wrapper keeps the full frozen matrix as-is (zero
    additional optimizer state -- requires_grad=False) and adds a
    [n_special, hidden] trainable table (a few thousand params total) that
    supplies output only for those specific ids.

    IMPORTANT ordering requirement: apply this AFTER apply_lora(), not
    before. get_lora_target_modules() auto-detects by walking
    named_modules() for nn.Linear leaves; if this wrapper (or its internal
    `special_embedding`) existed before LoRA wrapping, nothing bad happens
    for the Embedding itself (LoRA only targets nn.Linear), but wrapping
    order still matters for the lm_head twin of this class below, where the
    internal base_lm_head Linear WOULD get incorrectly auto-detected and
    LoRA-wrapped if this patch ran first. Keep both patches after LoRA for
    consistency.
    """

    def __init__(self, base_embedding, special_token_ids, init_from_id=None):
        super().__init__()
        self.base_embedding = base_embedding
        for p in self.base_embedding.parameters():
            p.requires_grad = False

        self.special_token_ids = list(special_token_ids)
        self._id_to_local = {tok_id: i for i, tok_id in enumerate(self.special_token_ids)}

        hidden_size = base_embedding.embedding_dim
        self.special_embedding = nn.Embedding(len(self.special_token_ids), hidden_size)
        # base_embedding.weight lives wherever device_map="auto" put this
        # shard (not necessarily cuda:0, and not necessarily the same
        # device as other layers) -- a freshly-constructed nn.Embedding
        # defaults to CPU/fp32 regardless, so without this it silently ends
        # up on a different device than the input_ids it's indexed with,
        # and F.embedding hard-errors on device mismatch instead of
        # auto-transferring.
        self.special_embedding = self.special_embedding.to(
            device=base_embedding.weight.device, dtype=base_embedding.weight.dtype
        )

        if init_from_id is not None:
            with torch.no_grad():
                anchor_row = base_embedding.weight[init_from_id].detach().clone()
                self.special_embedding.weight.copy_(
                    anchor_row.unsqueeze(0).repeat(len(self.special_token_ids), 1)
                )

    def forward(self, input_ids):
        out = self.base_embedding(input_ids)
        # Self-heal device placement here rather than trusting the device
        # captured at __init__ time. With device_map="auto", accelerate's
        # dispatch hooks can finalize where a shard's weights actually live
        # AFTER this module was constructed (observed in practice: the
        # __init__-time snapshot of base_embedding.weight.device ended up
        # stale, special_embedding stayed on the device recorded at patch
        # time while base_embedding got moved to a different GPU by the
        # time forward actually ran). `out` here is guaranteed correct --
        # it's literally the tensor produced by the call that just
        # succeeded above -- so use its device as ground truth instead of
        # any earlier assumption.
        if self.special_embedding.weight.device != out.device or self.special_embedding.weight.dtype != out.dtype:
            self.special_embedding = self.special_embedding.to(device=out.device, dtype=out.dtype)
        out = out.clone()
        for tok_id in self.special_token_ids:
            mask = input_ids == tok_id
            if not mask.any():
                continue
            local_idx = self._id_to_local[tok_id]
            vec = self.special_embedding(
                torch.tensor(local_idx, device=input_ids.device)
            )
            out[mask] = vec
        return out

    # Proxied in case any HF/Coconut code reaches for these attributes
    # directly on what it thinks is a plain nn.Embedding.
    @property
    def weight(self):
        return self.base_embedding.weight

    @property
    def embedding_dim(self):
        return self.base_embedding.embedding_dim

    @property
    def num_embeddings(self):
        return self.base_embedding.num_embeddings


class SparseTrainableLMHeadPatch(nn.Module):
    """
    Output-side twin of SparseTrainableEmbeddingPatch -- see that class's
    docstring for the full rationale. Keeps lm_head's full
    [vocab, hidden] weight frozen; adds a small trainable
    [n_special, hidden] linear supplying logits only for the specific
    output token ids that need to learn to be predicted.
    """

    def __init__(self, base_lm_head, special_token_ids, init_from_id=None):
        super().__init__()
        self.base_lm_head = base_lm_head
        for p in self.base_lm_head.parameters():
            p.requires_grad = False

        self.special_token_ids = list(special_token_ids)
        hidden_size = base_lm_head.in_features
        self.special_lm_head = nn.Linear(hidden_size, len(self.special_token_ids), bias=False)
        # See SparseTrainableEmbeddingPatch's identical fix above -- same
        # device_map="auto" issue applies to lm_head's shard too.
        self.special_lm_head = self.special_lm_head.to(
            device=base_lm_head.weight.device, dtype=base_lm_head.weight.dtype
        )

        if init_from_id is not None:
            with torch.no_grad():
                anchor_row = base_lm_head.weight[init_from_id].detach().clone()
                self.special_lm_head.weight.copy_(
                    anchor_row.unsqueeze(0).repeat(len(self.special_token_ids), 1)
                )

    def forward(self, hidden_states):
        out = self.base_lm_head(hidden_states)
        # Same self-heal as SparseTrainableEmbeddingPatch.forward above --
        # don't trust the device captured at __init__ time, use the device
        # of the tensor that just successfully passed through base_lm_head.
        if self.special_lm_head.weight.device != hidden_states.device:
            self.special_lm_head = self.special_lm_head.to(hidden_states.device)
        out = out.clone()  # see SparseTrainableEmbeddingPatch.forward for why
        extra_logits = self.special_lm_head(hidden_states)
        for i, tok_id in enumerate(self.special_token_ids):
            out[..., tok_id] = extra_logits[..., i]
        return out

    @property
    def weight(self):
        return self.base_lm_head.weight

    @property
    def in_features(self):
        return self.base_lm_head.in_features

    @property
    def out_features(self):
        return self.base_lm_head.out_features


def apply_sparse_new_token_patch(model, special_token_ids, init_from_id=None):
    """
    Replace `model`'s input embedding and lm_head with the sparse trainable
    wrappers above, scoped to special_token_ids.

    Call this AFTER apply_lora() and AFTER model.resize_token_embeddings()
    -- resize first so the base embedding/lm_head are already the right
    vocab size (their frozen weight just needs to physically have rows at
    those ids, even though this patch overrides what's returned for them);
    LoRA-wrap first so target-module auto-detection never sees these
    wrapper's internal Linear/Embedding submodules and never gets tempted
    to LoRA-adapt them.
    """
    base = _unwrap_to_base(model)

    input_embeddings = base.get_input_embeddings()
    patched_embed = SparseTrainableEmbeddingPatch(
        input_embeddings, special_token_ids, init_from_id
    )
    base.set_input_embeddings(patched_embed)

    lm_head = base.lm_head
    patched_head = SparseTrainableLMHeadPatch(lm_head, special_token_ids, init_from_id)
    base.lm_head = patched_head

    n_trainable = sum(
        p.numel()
        for p in list(patched_embed.special_embedding.parameters())
        + list(patched_head.special_lm_head.parameters())
    )
    print(
        f"Sparse new-token patch applied: {len(special_token_ids)} special "
        f"token ids, {n_trainable:,} new trainable params "
        f"(vs. {input_embeddings.num_embeddings * input_embeddings.embedding_dim * 2:,} "
        f"if embed_tokens+lm_head were fully unfrozen instead)."
    )
    return model
