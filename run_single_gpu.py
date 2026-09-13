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
from lora_utils import apply_lora, get_trainable_state_dict
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


def main():

    parser = argparse.ArgumentParser(description="coconut-single-gpu")
    parser.add_argument("config_file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # NOTE: `device` above is now only a fallback/reference value (e.g. for
    # torch.load map_location) -- the model itself is NOT placed on it as a
    # single device once device_map="auto" is used below. See input_device.

    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    print("Config:", config_dict)

    configs = Config(config_dict)
    set_seed(configs.seed)
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

        else:
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

    if not (configs.cot or configs.no_thoughts or configs.no_cot):
        model.resize_token_embeddings(len(tokenizer))
        embeddings = model.get_input_embeddings()
        target_id = tokenizer.convert_tokens_to_ids("<<")
        for token_id in [latent_id, start_id, end_id]:
            target_embedding = embeddings.weight.data[target_id]
            embeddings.weight.data[token_id] = target_embedding
            lm_head = model.lm_head
            lm_head.weight.data[token_id] = lm_head.weight.data[target_id]

    if configs.no_thoughts:
        configs.c_thought = 0
        configs.coconut = False

    # LoRA wrap happens here: after resize_token_embeddings/new-token init
    # above (so the resized embedding matrix's raw weights get copied in
    # before anything is frozen), and before Coconut wraps the model (so
    # Coconut's base_causallm is the PeftModel -- Coconut just proxies
    # forward()/get_input_embeddings() calls through, doesn't care whether
    # the object underneath is a raw AutoModelForCausalLM or a PeftModel).
    # Default on; set use_lora: false in the yaml config to fall back to the
    # old full-finetune path (e.g. for A/B-testing whether LoRA changes
    # anything besides memory).
    if getattr(configs, "use_lora", True):
        model = apply_lora(model, configs)
    else:
        print("use_lora=false in config -- running full finetune (all params trainable).")

    if configs.coconut:
        model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)

    if configs.load_model_path != "None" and not loaded:
        print(model.load_state_dict(saved_weights, strict=False))

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
                loss.backward()

                if (step + 1) % configs.gradient_accumulation_steps == 0 or step == len(
                    train_dataloader
                ) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar.update(1)

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
