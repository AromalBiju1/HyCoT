# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Single-GPU strip of Meta's original run.py for Kaggle T4 (no NCCL/FSDP/DDP).
# Removed: dist.init_process_group, LOCAL_RANK/RANK/WORLD_SIZE, FSDP/DDP wrap,
# DistributedSampler, dist.barrier/all_reduce. Kept: full training/eval logic,
# checkpoint resume, wandb logging, gradient_accumulation_steps.
#
# NOTE: batch_size_training in your config MUST be 1 (Coconut.forward asserts
# this for the Qwen3.5 patch). Use gradient_accumulation_steps for effective
# batch size instead.
#
# DUAL-T4 SHARDING: model is loaded with device_map="auto" (accelerate shards
# layers across both visible GPUs), same as smoke_test_coconut.py, because the
# model + autograd graph OOM'd on a single T4 (14.53/14.56 GiB during
# backward). This means model.parameters() live on more than one device, so:
#   - we never call model.to(device) or model.to(bfloat16) after load (that
#     would collapse the sharding back onto one device) -- dtype is set via
#     torch_dtype at from_pretrained time instead.
#   - every batch dict must be moved to wherever the *input* embeddings
#     landed (input_device below), not a single hardcoded `device`. Accelerate
#     moves activations between shards internally as they cross layer
#     boundaries; only the initial input needs to start on the right device.
#   - IMPORTANT: this is model-parallelism, not data-parallelism. Do not wrap
#     this in DDP/data-parallel for multi-sample throughput without separate
#     work -- the two don't compose for free.

# Reduce CUDA allocator fragmentation -- MUST be set before torch is imported
# (before any CUDA context exists). The last OOM traceback showed real memory
# lost to fragmentation (reserved-but-unallocated > 1GiB), on top of the
# genuine memory pressure from running a ~4B model on 2x14.56GiB T4s.
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer


def make_optimizer(model, configs):
    # Pre-LoRA this had to reach for bitsandbytes' PagedAdamW8bit, because
    # vanilla AdamW's fp32 optimizer state (~8 bytes/param) on a full ~4B-
    # param finetune is ~32GB -- doesn't fit on 2x14.56GiB T4 alongside
    # weights+grads. With LoRA, model.parameters() with requires_grad=True
    # is now only the adapter matrices (megabytes, not billions of params),
    # so that whole memory wall is gone -- optimizer state for LoRA params
    # is negligible regardless of precision. Plain AdamW is the right
    # default now; the 8-bit path is kept only as an explicit opt-in in case
    # you later unfreeze something big (e.g. the MLP bridge, if it's large)
    # and want the memory headroom back.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"Optimizer covers {n_trainable:,} trainable params "
        f"out of {n_total:,} total ({100 * n_trainable / n_total:.3f}%)."
    )
    if n_trainable == 0:
        raise ValueError(
            "No trainable parameters found -- LoRA wrap likely didn't run, or "
            "target_modules matched nothing. Check apply_lora() logs above "
            "for the detected target_modules list before training."
        )

    if getattr(configs, "force_8bit_optimizer", False):
        import bitsandbytes as bnb
        print("Using bitsandbytes PagedAdamW8bit (forced via config: force_8bit_optimizer).")
        return bnb.optim.PagedAdamW8bit(
            trainable_params, lr=configs.lr, weight_decay=configs.weight_decay
        )

    return optim.AdamW(
        trainable_params, lr=configs.lr, weight_decay=configs.weight_decay
    )

import wandb

from coconut import Coconut
from lora_utils import apply_lora, apply_sparse_new_token_patch, get_trainable_state_dict
from dataset import (
    get_dataset,
    get_question_latent_dataset,
    get_cot_latent_dataset,
    MyCollator,
)

from tqdm import tqdm
from copy import copy
import os, sys
import yaml
import json
import gc
import argparse
from utils import Config, set_seed


def _parse_cli_overrides(unknown_args):
    """
    Parse any number of --key value (or --key=value) pairs into a dict.

    Why this exists instead of parser.add_argument("--clip-grad-norm", ...)
    per config key: this project's yaml config keeps growing new keys
    (clip_grad_norm, save_every_n_steps, use_lora, lora_r,
    lora_target_modules, train_subset_size, ... and whatever's added next)
    as it's iterated on. Hardcoding a flag per key means editing this
    parser every single time one gets added -- fragile busywork that will
    silently drift out of sync (add a new yaml key, forget to also add its
    --flag here, CLI override for it quietly does nothing). Accepting
    arbitrary --key value instead makes every current AND future config key
    overridable with zero changes to this file, ever.

    Values are parsed with yaml.safe_load so "true"/"500"/"1.0e-4" come
    through as bool/int/float instead of always landing as the string
    "true"/"500"/"1.0e-4" -- matters because Config's downstream code does
    e.g. `if configs.debug:` and `configs.lr * ...`, not string comparisons.
    """
    overrides = {}
    i = 0
    while i < len(unknown_args):
        token = unknown_args[i]
        if not token.startswith("--"):
            raise ValueError(
                f"Unrecognized argument (expected --key value or --key=value): {token!r}"
            )
        key = token[2:]
        if "=" in key:
            key, value_str = key.split("=", 1)
        else:
            if i + 1 >= len(unknown_args):
                raise ValueError(f"--{key} given with no value")
            value_str = unknown_args[i + 1]
            i += 1
        overrides[key] = yaml.safe_load(value_str)
        i += 1
    return overrides


def main():

    parser = argparse.ArgumentParser(description="coconut-single-gpu")
    parser.add_argument("config_file")
    # parse_known_args (not parse_args): anything not matched above (every
    # --key value pair) is collected in `unknown` and handled by
    # _parse_cli_overrides, instead of argparse rejecting it as unrecognized.
    args, unknown = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # NOTE: `device` above is now only a fallback/reference value (e.g. for
    # torch.load map_location) -- the model itself is NOT placed on it as a
    # single device once device_map="auto" is used below. See input_device.

    # Disable cuDNN convolutions: this model's DeltaNet layers run a large
    # grouped conv1d (groups=8192, effectively depthwise) as part of every
    # forward pass. On T4 (Turing, sm_75 -- predates native BF16 tensor-core
    # support, which arrived with Ampere/sm_80), cuDNN's algorithm search for
    # this specific op under bf16 can fail to find ANY valid engine for
    # certain input shapes ("GET was unable to find an engine to execute
    # this computation"). Observed in practice: ran clean for 3 full epochs
    # (curriculum stage 0, zero latent tokens -- see epochs_per_stage), then
    # crashed on the very first batch of the epoch where the curriculum
    # first inserts real <|latent|> tokens (stage 1) -- a shape/pattern
    # cuDNN's heuristic search hadn't been asked to handle before. Disabling
    # cuDNN for convs forces PyTorch's plain (non-cuDNN) conv1d path, which
    # doesn't have this failure mode -- a real fix for the crash, not a
    # retry-and-hope. Cost: conv1d becomes somewhat slower, but this project
    # is already on slow reference kernels everywhere (no fla/causal_conv1d
    # installed) so the relative overhead here is small. Set
    # disable_cudnn_conv: false in the yaml if you later move to Ampere+
    # hardware and want to re-enable it.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    cli_overrides = _parse_cli_overrides(unknown)
    if cli_overrides:
        print(f"CLI overrides applied on top of {args.config_file}: {cli_overrides}")
        config_dict.update(cli_overrides)

    print("Config:", config_dict)

    configs = Config(config_dict)
    set_seed(configs.seed)

    disable_cudnn_conv = getattr(configs, "disable_cudnn_conv", True)
    if disable_cudnn_conv:
        torch.backends.cudnn.enabled = False
        print("cuDNN disabled globally (T4 depthwise-conv engine-search workaround).")
    save_dir = os.path.join(configs.save_path, configs.name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cur_ckpts = os.listdir(save_dir)

    # resume-after-preemption logic (unchanged from original, no dist needed)
    if len(cur_ckpts) > 0 and not configs.only_eval:
        print(
            "Warning: found previous run and gonna resume from that. "
            "the inputted `resume` argument is ignored!"
        )
        # startswith("checkpoint_") intentionally does NOT match this
        # script's periodic_checkpoint_/nan_checkpoint_/nangrad_checkpoint_
        # files (mid-epoch saves, see the training loop below) -- those use
        # different prefixes specifically so they're invisible to this
        # epoch-based auto-resume, since the sort below assumes every
        # matched file is "checkpoint_<integer epoch number>" and would
        # crash on "checkpoint_epoch3_step750" otherwise. A mid-epoch save
        # is for manually recovering from a killed session (load it by
        # path, it's just a state dict), not something this loop should
        # auto-pick-up as "epoch N is done."
        checkpoints = [f for f in cur_ckpts if f.startswith("checkpoint_")]
        checkpoints.sort(key=lambda x: int(x.split("_")[1]))
        latest_checkpoint = checkpoints[-1] if checkpoints else None
        configs.resume = int(latest_checkpoint.split("_")[1])
        load_dir = os.path.join(configs.save_path, configs.name, latest_checkpoint)
        configs.load_model_path = load_dir
        print(f"Loading from previous run epoch_{configs.resume}!")

    elif configs.resume != 0:
        if configs.load_model_path == "None":
            print(
                f"Warning: you want to skip the first {configs.resume} but you "
                "are not loading any existing checkpoint!"
            )
        print(
            f"Loading from {configs.load_model_path} and skip the first "
            f"{configs.resume} epochs"
        )

    model = AutoModelForCausalLM.from_pretrained(
        configs.model_id,
        torch_dtype=torch.bfloat16 if configs.bf16 else None,
        device_map="auto",
    )
    # dtype is now set at load time via torch_dtype (bf16 if configured) --
    # do NOT call model.to(torch.bfloat16) later, since that would require
    # collapsing the accelerate-sharded model back onto one device first.

    # Capture input_device HERE, on the raw AutoModelForCausalLM, before any
    # Coconut wrapping below -- Coconut doesn't proxy get_input_embeddings(),
    # so calling this after `model = Coconut(model, ...)` raises AttributeError.
    input_device = next(model.get_input_embeddings().parameters()).device

    # Gradient checkpointing: without flash-linear-attention (removed earlier --
    # its fused_recurrent kernel has no backward), both DeltaNet paths run on
    # plain PyTorch reference kernels, which materialize far more intermediate
    # activation tensors during backward than fla's chunked/fused kernels would.
    # This showed up as a real OOM inside loss.backward() (not optimizer.step(),
    # already fixed via bitsandbytes) on a longer-than-average GSM8K example a
    # few steps into training. Checkpointing recomputes activations during
    # backward instead of storing them all -- the standard fix for this failure
    # mode, as opposed to shrinking batch size further (already at the floor:
    # batch_size_training=1) or truncating sequences (a data hack, not a fix).
    # MUST happen here, on the raw AutoModelForCausalLM, before Coconut wraps it
    # below -- same ordering trap as input_device: Coconut doesn't proxy
    # gradient_checkpointing_enable() either.
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    # Required when the backbone is frozen (LoRA) + gradient checkpointing is
    # on: checkpointing recomputes activations during backward starting from
    # the recorded input, but if that input itself has requires_grad=False
    # (true for embed_tokens' output once the backbone is frozen), autograd
    # has nothing to hook the recomputed graph onto and silently produces no
    # gradient at all for anything downstream. This forces the input
    # embeddings' output to require grad regardless of the embedding layer's
    # own frozen weights, which is exactly what checkpointing needs here.
    model.enable_input_require_grads()
    print("Gradient checkpointing enabled (trades compute for activation memory).")

    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    loaded = False

    if configs.load_model_path != "None":
        # map_location="cpu": with device_map="auto" already sharding the live
        # model across GPUs, loading the checkpoint straight to a single
        # `device` here would both fight that placement and temporarily
        # double GPU memory. load_state_dict copies values into the existing
        # (already correctly placed) parameters regardless of the state
        # dict's own device, so cpu is the safe, low-memory choice.
        saved_weights = torch.load(configs.load_model_path, map_location="cpu")

        if configs.coconut and not any(
            k.startswith("base_causallm") for k in saved_weights.keys()
        ):
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

        elif not configs.coconut and any(
            k.startswith("base_causallm") for k in saved_weights.keys()
        ):
            raise ValueError("Cannot load coconut model weights into a causallm model")

        elif configs.coconut and any(
            k.startswith("base_causallm") for k in saved_weights.keys()
        ):
            pass  # resuming a preempted coconut run, handled below

        elif not configs.coconut:
            # Non-Coconut path: defer loading until AFTER LoRA wrapping below.
            # If we load now into the raw AutoModelForCausalLM, then wrap
            # with get_peft_model, the adapter keys (lora_A/lora_B) in the
            # checkpoint get rejected as unexpected_keys and the adapters
            # end up zero-initialized — silent failure.
            pass

        else:
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

    special_token_ids = []
    anchor_id = None
    if not (configs.cot or configs.no_thoughts or configs.no_cot):
        model.resize_token_embeddings(len(tokenizer))
        anchor_id = tokenizer.convert_tokens_to_ids("<<")
        special_token_ids = [latent_id, start_id, end_id]
        # NOTE: no longer manually copying embeddings.weight.data[token_id] /
        # lm_head.weight.data[token_id] from "<<" here -- that copy is now
        # done inside apply_sparse_new_token_patch() below (init_from_id),
        # AFTER LoRA wrapping, as part of setting up the sparse trainable
        # table for these 3 ids. Doing it here first would just be
        # overwritten/ignored once the sparse patch takes over forward() for
        # these ids anyway.

    if configs.no_thoughts:
        configs.c_thought = 0
        configs.coconut = False

    # LoRA wrap happens here: after resize_token_embeddings above (so the
    # resized embedding matrix already has the right vocab size), and
    # before Coconut wraps the model (so Coconut's base_causallm is the
    # PeftModel -- Coconut just proxies forward()/get_input_embeddings()
    # calls through, doesn't care whether the object underneath is a raw
    # AutoModelForCausalLM or a PeftModel).
    # Default on; set use_lora: false in the yaml config to fall back to the
    # old full-finetune path (e.g. for A/B-testing whether LoRA changes
    # anything besides memory).
    use_lora = getattr(configs, "use_lora", True)
    if use_lora:
        model = apply_lora(model, configs)
    else:
        print("use_lora=false in config -- running full finetune (all params trainable).")

    # Sparse new-token patch: replaces the earlier `lora_modules_to_save:
    # embed_tokens,lm_head` approach, which unfroze the full ~635M-param
    # embed_tokens/lm_head matrices just to let 3 rows move -- that's what
    # OOM'd (~10GB of AdamW state for 1.27B trainable params on a single
    # T4). This gives only those 3 rows their own small trainable table
    # (a few thousand params) and leaves the rest of embed_tokens/lm_head
    # frozen. Only meaningful under LoRA -- if use_lora=false you're doing
    # a full finetune anyway and embed_tokens/lm_head are already fully
    # trainable as part of that, so skip this (it would wrongly freeze them).
    if use_lora and special_token_ids:
        model = apply_sparse_new_token_patch(model, special_token_ids, anchor_id)

    if configs.coconut:
        model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)

    if configs.load_model_path != "None" and not loaded:
        result = model.load_state_dict(saved_weights, strict=False)
        print(f"load_state_dict: unexpected_keys={len(result.unexpected_keys)}, missing_keys={len(result.missing_keys)}")
        if result.unexpected_keys:
            print(f"  unexpected_keys sample: {result.unexpected_keys[:5]}")
        # Verify LoRA B weights loaded correctly (must be non-zero after training)
        lora_b_sum = 0
        lora_b_count = 0
        for name, param in model.named_parameters():
            if "lora_B" in name or "lora_b" in name:
                lora_b_sum += param.data.abs().sum().item()
                lora_b_count += 1
        print(f"  lora_B params: {lora_b_count}, abs_sum={lora_b_sum:.6f} (must be >0 for trained adapter)")

    # input_device and gradient checkpointing were both set up earlier, right
    # after the raw AutoModelForCausalLM load, before Coconut wrapping -- see
    # comments above (Coconut proxies neither get_input_embeddings() nor
    # gradient_checkpointing_enable()).
    print(f"Sharded model loaded via device_map='auto'; input_device={input_device}")
    print(model)

    question_val = [d["question"] for d in json.load(open(configs.val_path))]
    answers_val = [
        d["answer"].replace(",", "").strip() for d in json.load(open(configs.val_path))
    ]
    cot_val = ["\n".join(d["steps"]) for d in json.load(open(configs.val_path))]

    base_dataset_valid = get_dataset(
        configs.val_path, tokenizer, max_size=32 if configs.debug else 100000000
    )

    if not configs.only_eval:
        base_dataset_train = get_dataset(
            configs.train_path, tokenizer, max_size=5000 if configs.debug else 100000000
        )

    max_new_tokens = 64 if "gsm" in configs.val_path else 128

    total_train_steps = 0

    if getattr(configs, "wandb_mode", None):
        os.environ["WANDB_MODE"] = configs.wandb_mode
    if not configs.debug and not configs.only_eval:
        wandb_run = wandb.init(project=configs.project, name=configs.name)
        wandb_run.config.update(configs, allow_val_change=True)
        text_table = wandb.Table(columns=["step", "text"])
    else:
        wandb_run = None

    if configs.reset_optimizer:
        optimizer = None
    else:
        optimizer = make_optimizer(model, configs)

    best_acc = 0

    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)

    print("cudnn.enabled just before training:", torch.backends.cudnn.enabled)

    for epoch in range(configs.resume, configs.num_epochs):

        scheduled_stage = (
            0 if (configs.cot or configs.no_cot) else epoch // configs.epochs_per_stage
        )
        dataset_gen_val = get_question_latent_dataset(
            scheduled_stage,
            base_dataset_valid,
            configs,
            start_id,
            latent_id,
            end_id,
            no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
        )

        # eval_max_examples caps ONLY this generation-based accuracy set, not
        # the cheap forward-pass "eval loss" above -- the whole cost problem
        # is model.generate() (autoregressive, ~15-23s/example observed),
        # not the loss computation (single forward pass, negligible). Full
        # val set (~500 examples here) x ~18s/example x every epoch is the
        # actual 2+ hour/epoch bottleneck; capping this to e.g. 50 gives a
        # real accuracy signal in minutes instead of hours, at the cost of
        # a noisier estimate. Unset (None) keeps the full original behavior.
        eval_max_examples = getattr(configs, "eval_max_examples", None)
        if eval_max_examples and len(dataset_gen_val) > eval_max_examples:
            dataset_gen_val = dataset_gen_val.select(range(eval_max_examples))

        # sequential sampling (no DistributedSampler needed for a single device)
        valid_gen_dataloader = torch.utils.data.DataLoader(
            dataset_gen_val,
            num_workers=1,
            pin_memory=True,
            batch_size=1,
            collate_fn=collator,
            shuffle=False,
        )

        if not configs.only_eval:

            dataset_train = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_train,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
                shuffle=True,
            )

            # NOTE: batch_size_training must be 1 -- Coconut.forward asserts
            # this. Use gradient_accumulation_steps for effective batch size.
            train_dataloader = torch.utils.data.DataLoader(
                dataset_train,
                num_workers=1,
                shuffle=True,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
            )

            dataset_loss_val = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_valid,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
            )

            valid_loss_dataloader = torch.utils.data.DataLoader(
                dataset_loss_val,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
            )

            if configs.reset_optimizer:
                del optimizer
                optimizer = make_optimizer(model, configs)

            # Restore use_cache=False for training + gradient checkpointing --
            # the previous epoch's generation loop (if any) turned it back on.
            getattr(model, "base_causallm", model).config.use_cache = False

            model.train()

            total_length = len(train_dataloader) // configs.gradient_accumulation_steps
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch+1}",
                total=total_length,
                dynamic_ncols=True,
            )

            for step, batch in enumerate(train_dataloader):

                if step == 0 and wandb_run:
                    print("logging training data")
                    cur_bs = len(batch["input_ids"])
                    text_str = ""
                    for data_idx in range(cur_bs):
                        for token_idx in range(len(batch["input_ids"][data_idx])):
                            text_str += (
                                str(batch["input_ids"][data_idx][token_idx].item())
                                + " "
                                + str(batch["labels"][data_idx][token_idx].item())
                                + " "
                                + tokenizer.decode(batch["input_ids"][data_idx][token_idx])
                                + "\n"
                            )
                        text_str += "====" * 10 + "\n"
                    text_table.add_data(total_train_steps, text_str)
                    wandb_run.log({"data_table": copy(text_table)})

                total_train_steps += 1
                batch = {
                    key: batch[key].to(input_device)
                    for key in batch.keys()
                    if key != "idx"
                }

                outputs = model(**batch)

                loss = outputs.loss / configs.gradient_accumulation_steps

                # NaN/Inf guard: with a run this long (reference kernels,
                # no fla/causal_conv1d -- confirmed slow path from the
                # transformers warnings -- likely multi-day for the full
                # 25-epoch curriculum), a NaN partway through would
                # otherwise train silently on garbage for however long is
                # left, and you'd only discover it at the final accuracy
                # readout with no way to tell which step it started at or
                # recover the last good state. Catch it the moment it
                # appears instead: save what's trained so far under a
                # clearly-labeled emergency checkpoint and stop cleanly,
                # rather than continuing to update on Inf/NaN gradients or
                # (worse) silently saving over a good epoch checkpoint at
                # the end of this corrupted epoch.
                if not torch.isfinite(loss):
                    print(
                        f"\n!!! NaN/Inf LOSS at epoch {epoch+1}, step {step} "
                        f"(total_train_steps={total_train_steps}): loss={loss.item()}. "
                        f"Stopping and saving emergency checkpoint -- see "
                        f"nan_checkpoint_epoch{epoch+1}_step{total_train_steps} "
                        f"in {save_dir}."
                    )
                    torch.save(
                        get_trainable_state_dict(model),
                        os.path.join(
                            save_dir,
                            f"nan_checkpoint_epoch{epoch+1}_step{total_train_steps}",
                        ),
                    )
                    raise SystemExit(
                        1
                    ) from None  # deliberate hard stop, not a bug to catch upstream

                loss.backward()

                if (step + 1) % configs.gradient_accumulation_steps == 0 or step == len(
                    train_dataloader
                ) - 1:
                    # Same reasoning as the loss check above, but for
                    # gradients specifically: a finite loss can still
                    # produce an Inf/NaN gradient (e.g. a genuine numerical
                    # blowup inside backward, distinct from the forward
                    # pass itself going bad). Checking right before the
                    # optimizer step -- not every micro-step -- keeps this
                    # cheap under gradient_accumulation_steps > 1, since
                    # it's only the moment the update would actually happen.
                    bad_grad_found = False
                    for name, param in model.named_parameters():
                        if param.grad is not None and not torch.isfinite(param.grad).all():
                            bad_grad_found = True
                            print(f"NaN/Inf GRADIENT in {name}")
                    if bad_grad_found:
                        print(
                            f"\n!!! NaN/Inf GRADIENT at epoch {epoch+1}, step {step} "
                            f"(total_train_steps={total_train_steps}). Stopping and "
                            f"saving emergency checkpoint before optimizer.step() runs "
                            f"on corrupted gradients."
                        )
                        torch.save(
                            get_trainable_state_dict(model),
                            os.path.join(
                                save_dir,
                                f"nangrad_checkpoint_epoch{epoch+1}_step{total_train_steps}",
                            ),
                        )
                        raise SystemExit(1) from None

                    # Gradient clipping: yesterday's full-finetune run went
                    # NaN via exponential loss growth -- the classic
                    # signature of an unclipped gradient blowup (one large
                    # update produces a worse next step, compounding). The
                    # NaN/Inf checks above catch that AFTER it happens and
                    # stop cleanly, but don't prevent it. LoRA's much
                    # smaller update magnitude makes a repeat less likely,
                    # but "less likely" isn't "fixed" -- clipping is the
                    # actual preventive lever, not a guess. max_norm=1.0 is
                    # a standard default (configurable via clip_grad_norm in
                    # the yaml); set clip_grad_norm: 0 to disable.
                    clip_grad_norm = getattr(configs, "clip_grad_norm", 1.0)
                    if clip_grad_norm and clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad],
                            max_norm=clip_grad_norm,
                        )

                    optimizer.step()
                    optimizer.zero_grad()
                    pbar.update(1)

                # Periodic mid-epoch checkpoint: end-of-epoch save alone
                # means a Kaggle session timeout (~9-12h cap) partway
                # through a 5000-step epoch loses the entire epoch's
                # progress. save_every_n_steps defaults to 250 (a save is
                # just the ~32M trainable params, cheap) -- set to 0 in the
                # config to disable.
                save_every_n_steps = getattr(configs, "save_every_n_steps", 250)
                if (
                    save_every_n_steps
                    and not configs.debug
                    and not configs.only_eval
                    and total_train_steps % save_every_n_steps == 0
                ):
                    torch.save(
                        get_trainable_state_dict(model),
                        os.path.join(
                            save_dir,
                            f"periodic_checkpoint_epoch{epoch+1}_step{total_train_steps}",
                        ),
                    )

                if wandb_run:
                    log_dict = {
                        "train/epoch": epoch + 1,
                        "train/step": epoch * len(train_dataloader) + step,
                        "train/loss": loss.detach().float()
                        * configs.gradient_accumulation_steps,
                    }
                    wandb_run.log(log_dict)

                pbar.set_description(
                    f"Training Epoch: {epoch+1}/{configs.num_epochs}, batch {step}/{len(train_dataloader)} "
                    f"completed (loss: {round(float(loss.detach().float() * configs.gradient_accumulation_steps), 4)}"
                )
            pbar.close()

            if not configs.save_only_improve and not configs.debug and not configs.only_eval:
                # trainable-only save: with LoRA this is the adapter matrices
                # (megabytes), not the full frozen 4B backbone (gigabytes).
                # See get_trainable_state_dict() docstring for why plain
                # model.state_dict() would silently defeat the point of LoRA.
                torch.save(
                    get_trainable_state_dict(model),
                    os.path.join(save_dir, f"checkpoint_{epoch + 1}"),
                )
                print("saving model (trainable params only).")
                gc.collect()
                torch.cuda.empty_cache()

            # val loss
            total_loss = 0
            with torch.no_grad():
                model.eval()
                for step, batch in enumerate(valid_loss_dataloader):
                    batch = {
                        key: batch[key].to(input_device)
                        for key in batch.keys()
                        if key != "idx"
                    }
                    outputs = model(**batch)
                    loss = outputs.loss
                    total_loss += loss.item()

                if wandb_run:
                    log_dict = {"eval/loss": total_loss / len(valid_loss_dataloader)}
                    wandb_run.log(log_dict)
                    print("eval loss", total_loss / len(valid_loss_dataloader))

        # eval_every_n_epochs gates ONLY the expensive generation-accuracy
        # loop below (the val-loss check above always runs -- it's cheap).
        # Default 1 preserves the original "every epoch" behavior. Always
        # run it on only_eval mode (that's the whole point of that mode) and
        # on the final epoch (so you always get a real final accuracy
        # number even if the interval didn't land on it).
        eval_every_n_epochs = getattr(configs, "eval_every_n_epochs", 1)
        is_final_epoch = (epoch + 1) == configs.num_epochs
        run_full_eval = (
            configs.only_eval
            or eval_every_n_epochs <= 1
            or (epoch + 1) % eval_every_n_epochs == 0
            or is_final_epoch
        )

        if not run_full_eval:
            print(
                f"Skipping generation-based eval this epoch (epoch {epoch+1}, "
                f"eval_every_n_epochs={eval_every_n_epochs}) -- val loss above "
                f"still ran."
            )
        else:
            # val generation accuracy
            total_length = len(valid_gen_dataloader)
            pbar = tqdm(
                colour="blue", desc="Test Accuracy", total=total_length, dynamic_ncols=True
            )
            cor, cor_cot, total = 0, 0, 0

            # use_cache was disabled for training (required alongside gradient
            # checkpointing) but generation is much faster with KV caching and
            # doesn't need the checkpointing memory tradeoff -- re-enable it just
            # for this generate() loop. model may be wrapped in Coconut (real
            # config lives at model.base_causallm.config), or be the plain
            # AutoModelForCausalLM if configs.coconut is False.
            gen_config = getattr(model, "base_causallm", model).config
            gen_config.use_cache = True

            with torch.no_grad():
                model.eval()
                for idx, batch in enumerate(valid_gen_dataloader):
                    test_idx = batch["idx"][0]

                    batch = {
                        k: v.to(input_device)
                        for k, v in batch.items()
                        if v is not None and k not in ["idx", "position_ids"]
                    }

                    assert len(batch["input_ids"]) == 1
                    answer = answers_val[test_idx.cpu().item()]
                    answer_cot = cot_val[test_idx.cpu().item()]
                    question = question_val[test_idx.cpu().item()]

                    total += 1

                    # synced_gpus was an FSDP requirement -- always False single-GPU
                    outputs = model.generate(
                        **batch, max_new_tokens=max_new_tokens, synced_gpus=False
                    )

                    text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    answer_output = text_output.split("#")[-1].replace(",", "").strip()
                    cot_output = ("\n".join(text_output.split("\n")[1:])).split("#")[0].strip()

                    if idx < 5:
                        print(f"Question {test_idx}: Answer = '{answer}' CoT = '{answer_cot}'")
                        print(f"Full output: '{tokenizer.decode(outputs[0])}'")
                        print(f"Extracted Output: '{answer_output}'")

                    cor += answer_output == answer
                    cor_cot += cot_output == answer_cot

                    pbar.update(1)
                    pbar.set_description(f"Test accuracy: {round(cor / total, 2)}")

                pbar.close()
                print(f"Cor={cor}, CoT={cor_cot}, Total={total}")

            print(f"Accuracy on validation set: {cor} / {total} = {cor/total}")
            print(f"CoT match on validation set: {cor_cot} / {total} = {cor_cot/total}")
            sys.stdout.flush()

            if wandb_run:
                wandb_run.log({"eval/acc": cor / total, "eval/cot_em": cor_cot / total})

            if configs.only_eval:
                break

            if cor / total > best_acc and configs.save_only_improve and not configs.debug and not configs.only_eval:
                torch.save(
                    get_trainable_state_dict(model),
                    os.path.join(save_dir, f"checkpoint_{epoch + 1}"),
                )
                print("saving model (trainable params only).")
                best_acc = cor / total
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()