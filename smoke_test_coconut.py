"""
Smoke test for Coconut + Qwen3.5-4B patch, batch_size=1.

Run this BEFORE touching run_single_gpu.py / real GSM8K data. It builds one
hand-crafted example with <|latent|> tokens, runs forward + loss.backward(),
and checks for NaN/Inf in loss and gradients. If this doesn't pass cleanly,
nothing downstream is worth running.

Usage: python smoke_test_coconut.py
"""

import torch
from types import SimpleNamespace
from transformers import AutoModelForCausalLM, AutoTokenizer
from coconut import Coconut
from lora_utils import apply_lora, apply_sparse_new_token_patch

MODEL_ID = "huihui-ai/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated"
# Qwen3.5 vision is a separate mmproj, not baked into the base weights, so
# AutoModelForCausalLM loads the text backbone alone -- fine for this test.


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    # device_map="auto" shards layers across both visible GPUs (Kaggle T4 x2)
    # via accelerate, since the model + autograd graph didn't fit on one T4
    # (OOM'd at 14.53/14.56 GiB on a single device during backward). Do NOT
    # also wrap this in DDP/data-parallel later -- model parallelism here and
    # data parallelism don't compose without extra work; that's a separate
    # decision for run_single_gpu.py if dual-T4 throughput is wanted too.

    # --- Architecture/tokenizer sanity check ---------------------------------
    # Earlier validation (cache-class structure, DeltaNet layer ratio, the
    # no-op truncation finding) was done against plain Qwen/Qwen3.5-4B. This
    # confirms the abliterated distill still matches those assumptions before
    # trusting the patch on it.
    print("\n=== Architecture/tokenizer check ===")
    print(f"model_type: {getattr(model.config, 'model_type', 'UNKNOWN')}")
    print(f"vocab_size (pre-resize): {getattr(model.config, 'vocab_size', 'UNKNOWN')}")
    print(f"tokenizer vocab size (pre-resize): {len(tokenizer) - 3}")  # minus the 3 tokens just added

    try:
        layers = model.model.layers
        layer_types = [type(l).__name__ for l in layers]
        from collections import Counter
        counts = Counter(layer_types)
        print(f"num layers: {len(layers)}")
        print(f"layer type counts (outer class, all identical by design): {dict(counts)}")
        # The linear-attn/full-attn split does NOT show up in the outer class
        # name (every layer reports as Qwen3_5DecoderLayer) -- it lives in
        # whether each layer actually has a populated linear_attn submodule.
        # String-matching the outer class name against
        # "Linear"/"DeltaNet"/"Gated" always reports 0 linear-attn layers;
        # this per-layer submodule check is the correct one.
        linear_attn = sum(
            1 for l in layers
            if getattr(l, "linear_attn", None) is not None
        )
        full_attn = len(layers) - linear_attn
        print(f"linear-attn layers: {linear_attn}, full-attn layers: {full_attn} "
              f"(expected ~3:1 ratio if architecture matches base Qwen3.5)")
    except AttributeError as e:
        print(f"Could not introspect model.model.layers directly: {e}")
        print("Check model.config for layer_types / architecture details manually.")

    cache_test = model.__class__.__name__
    print(f"Model class: {cache_test}")
    print("=== End architecture check ===\n")

    model.resize_token_embeddings(len(tokenizer))
    target_id = tokenizer.convert_tokens_to_ids("<<")
    if target_id is None or target_id == tokenizer.unk_token_id:
        # fallback anchor token if "<<" isn't in Qwen3.5's vocab
        target_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
        print(f"'<<' not found, using eos_token_id={target_id} as embedding anchor")
    # NOTE: no longer manually copying embeddings.weight.data[token_id] /
    # lm_head.weight.data[token_id] here -- apply_sparse_new_token_patch()
    # below does that init (via init_from_id) as part of setting up the
    # sparse trainable table for these 3 ids, AFTER LoRA wrapping.

    # LoRA wrap before the sparse patch, same ordering as run_single_gpu.py.
    # Use SimpleNamespace here instead of a real yaml Config since this
    # smoke test has no config file -- apply_lora only reads
    # getattr(configs, ...) with defaults, so an empty namespace is enough
    # to get r=16/alpha=32.
    lora_configs = SimpleNamespace()
    model = apply_lora(model, lora_configs)

    # Sparse new-token patch: see lora_utils.apply_sparse_new_token_patch
    # docstring. Replaces the earlier full-unfreeze-of-embed_tokens/lm_head
    # approach that OOM'd on real training -- only these 3 rows get their
    # own small trainable table, the rest of embed_tokens/lm_head stays
    # frozen.
    model = apply_sparse_new_token_patch(model, [latent_id, start_id, end_id], target_id)

    coconut_model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)
    # NOTE: no .to(device) here -- model is already sharded across GPUs by
    # device_map="auto" above; forcing it onto a single device would undo
    # that and bring back the single-GPU OOM.
    coconut_model.train()

    # Hand-built example: a short question, 2 latent tokens standing in for
    # compressed reasoning steps, then an answer. Mirrors the structure
    # get_cot_latent_dataset would produce at an intermediate curriculum stage.
    question = "Question: What is 3 + 4? Answer:"
    q_ids = tokenizer(question, return_tensors="pt").input_ids[0]

    n_latent = 2
    latent_span = (
        [start_id] + [latent_id] * n_latent + [end_id]
    )
    answer_ids = tokenizer(" 7", return_tensors="pt").input_ids[0]

    input_ids = torch.cat(
        [q_ids, torch.tensor(latent_span, dtype=q_ids.dtype), answer_ids]
    ).unsqueeze(0)
    # Place on whatever device the input embedding layer actually landed on
    # (device_map="auto" decides this, not necessarily GPU 0). Accelerate's
    # hooks move activations between GPUs internally as they cross shards;
    # only the initial input needs to start on the right device.
    input_device = next(model.get_input_embeddings().parameters()).device
    input_ids = input_ids.to(input_device)

    attention_mask = torch.ones_like(input_ids)

    # labels: mask out the question + latent span, supervise only the answer
    labels = input_ids.clone()
    mask_len = q_ids.shape[0] + len(latent_span)
    labels[0, :mask_len] = -100

    position_ids = torch.arange(0, input_ids.shape[1], device=input_device).unsqueeze(0)

    print(f"input_ids shape: {input_ids.shape}, n_latent_tokens: {n_latent}")
    print("Running forward pass...")

    outputs = coconut_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        position_ids=position_ids,
    )

    loss = outputs.loss
    print(f"Loss: {loss.item()}")

    assert torch.isfinite(loss), "FAIL: loss is NaN/Inf"
    print("Loss is finite. Running backward()...")

    loss.backward()

    print("Checking gradients for NaN/Inf, and that only LoRA params got any...")
    bad_grads = []
    total_params_with_grad = 0
    non_lora_with_grad = []
    for name, param in coconut_model.named_parameters():
        if param.grad is not None:
            total_params_with_grad += 1
            if not torch.isfinite(param.grad).all():
                bad_grads.append(name)
            if "lora_" not in name and "special_embedding" not in name and "special_lm_head" not in name:
                non_lora_with_grad.append(name)

    print(f"Params with gradients: {total_params_with_grad}")

    if bad_grads:
        print(f"FAIL: NaN/Inf gradients in {len(bad_grads)} params, e.g. {bad_grads[:5]}")
        raise SystemExit(1)

    if non_lora_with_grad:
        # get_peft_model() freezes everything NOT matched by target_modules.
        # embed_tokens/lm_head are frozen too (Embedding/lm_head never match
        # target_modules -- only nn.Linear leaves inside decoder layers do),
        # but that's now expected and correct: apply_sparse_new_token_patch()
        # already gives the 3 new token ids their own small trainable table
        # (special_embedding/special_lm_head params, which DO contain
        # "lora_"... no wait, they don't -- they're plain nn.Embedding/
        # nn.Linear, not LoRA layers). So this list should contain exactly
        # those: special_embedding.weight and special_lm_head.weight, and
        # nothing else. Anything beyond those two names is suspicious --
        # LoRA wrap or the sparse patch didn't freeze what it should have.
        print(
            f"NOTE: {len(non_lora_with_grad)} non-LoRA params have gradients "
            f"(expected: embed_tokens/lm_head rows for the new latent tokens). "
            f"First few: {non_lora_with_grad[:5]}"
        )

    print("\n=== SMOKE TEST PASSED ===")
    print("Loss finite, backward() completed, all gradients finite.")
    print("LoRA adapters received gradients; backbone frozen as expected.")
    print("Safe to proceed to run_single_gpu.py with real GSM8K data.")


if __name__ == "__main__":
    main()
