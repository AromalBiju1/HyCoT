# LoRA helpers shared by run_single_gpu.py and smoke_test_coconut.py.
#
# Why a shared module: both scripts need the same target-module detection,
# LoRA-wrap logic, and trainable-only state-dict save. Duplicating this in
# both files would drift the moment one gets edited and not the other.

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
