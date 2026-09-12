"""
Smoke test for Coconut + Qwen3.5-4B patch, batch_size=1.

Run this BEFORE touching run_single_gpu.py / real GSM8K data. It builds one
hand-crafted example with <|latent|> tokens, runs forward + loss.backward(),
and checks for NaN/Inf in loss and gradients. If this doesn't pass cleanly,
nothing downstream is worth running.

Usage: python smoke_test_coconut.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from coconut import Coconut

MODEL_ID = "huihui-ai/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated"
# Qwen3.5 vision is a separate mmproj, not baked into the base weights, so
# AutoModelForCausalLM loads the text backbone alone -- fine for this test.


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.autograd.set_detect_anomaly(True)

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
        MODEL_ID, torch_dtype=torch.bfloat16
    )

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
        print(f"layer type counts: {dict(counts)}")
        linear_attn = sum(v for k, v in counts.items() if "Linear" in k or "DeltaNet" in k or "Gated" in k)
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
    embeddings = model.get_input_embeddings()
    target_id = tokenizer.convert_tokens_to_ids("<<")
    if target_id is None or target_id == tokenizer.unk_token_id:
        # fallback anchor token if "<<" isn't in Qwen3.5's vocab
        target_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
        print(f"'<<' not found, using eos_token_id={target_id} as embedding anchor")
    for token_id in [latent_id, start_id, end_id]:
        embeddings.weight.data[token_id] = embeddings.weight.data[target_id]
        model.lm_head.weight.data[token_id] = model.lm_head.weight.data[target_id]

    coconut_model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)
    coconut_model = coconut_model.to(device)
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
    ).unsqueeze(0).to(device)

    attention_mask = torch.ones_like(input_ids)

    # labels: mask out the question + latent span, supervise only the answer
    labels = input_ids.clone()
    mask_len = q_ids.shape[0] + len(latent_span)
    labels[0, :mask_len] = -100

    position_ids = torch.arange(0, input_ids.shape[1], device=device).unsqueeze(0)

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

    print("Checking gradients for NaN/Inf...")
    bad_grads = []
    total_params_with_grad = 0
    for name, param in coconut_model.named_parameters():
        if param.grad is not None:
            total_params_with_grad += 1
            if not torch.isfinite(param.grad).all():
                bad_grads.append(name)

    print(f"Params with gradients: {total_params_with_grad}")

    if bad_grads:
        print(f"FAIL: NaN/Inf gradients in {len(bad_grads)} params, e.g. {bad_grads[:5]}")
        raise SystemExit(1)

    print("\n=== SMOKE TEST PASSED ===")
    print("Loss finite, backward() completed, all gradients finite.")
    print("Safe to proceed to run_single_gpu.py with real GSM8K data.")


if __name__ == "__main__":
    main()
