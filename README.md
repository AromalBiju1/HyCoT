# Coconut

The code base is the official implementation of [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769).

![coconut](assets/coconut.png)

> **This fork: Coconut on Qwen3.5 (gated DeltaNet hybrid).**
> Upstream Coconut was built and tested on GPT-2 / Llama-style pure-attention
> models. This fork ports it to Qwen3.5 (`huihui-ai/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated`),
> a **hybrid gated-DeltaNet + full-attention** architecture that upstream never
> covered. That architecture breaks several assumptions in the original code,
> so this fork carries Qwen3.5-specific fixes (all marked with `Qwen3.5 patch`
> / explanatory comments in the source):
>
> - **Cache is not sliceable KV.** Original `coconut.py` truncated
>   `past_key_values` per pass (`k[:, :, :n, :]`). Qwen3.5's cache holds
>   per-layer `LinearAttentionLayer` objects with `recurrent_states` /
>   `conv_states` that cannot be sliced like attention KV, so the cache is
>   passed through whole instead (`coconut.py`, `apply_qwen35_patch.py` keeps
>   the original one-shot patch for reference).
> - **In-place cache updates vs autograd.** Transformers' `LinearAttentionLayer`
>   mutates its state buffers with `.copy_()` (a CUDA-graph inference
>   optimization). That overwrites tensors autograd saved for backward, so any
>   `use_cache=True` + `.backward()` run fails. Fixed by monkey-patching both
>   update methods to reassign out-of-place, plus cloning DeltaNet states
>   between Coconut's re-entrant passes (`_patch_linear_attention_cache_for_training`,
>   `_clone_cache_states` in `coconut.py`).
> - **No KV cache under gradient checkpointing.** Checkpointing forces
>   `use_cache=False`, so `past_key_values` comes back `None` every pass and
>   the multi-pass loop used to `IndexError` on the first latent-token epoch.
>   Fixed with full-prefix recompute per pass (offset 0, new-slice-only logits)
>   when no cache is present (`coconut.py` `forward`).
> - **Single-process multi-GPU.** `device_map="auto"` model-parallel sharding
>   is used instead of DDP; dataset/distributed guards, device placement, and
>   checkpoint save/load are adapted accordingly (`dataset.py`,
>   `run_single_gpu.py`, `lora_utils.py`).
>
> Everything below the next section is upstream's documentation, kept for
> reference. The Qwen3.5 workflow (scripts, extra config keys, hardware notes)
> is documented in the **Qwen3.5 fork** section that follows it.
>
> ## Qwen3.5 fork usage
>
> ### Scripts
>
> - **`run_single_gpu.py`** — the training entry point for this fork (replaces
>   `torchrun ... run.py`). Single process, `device_map="auto"` sharding,
>   LoRA by default. Accepts CLI overrides for any config key:
>   `python run_single_gpu.py args/gsm_coconut.yaml --lr 5e-5 --debug true`.
> - **`smoke_test_coconut.py`** — fast pre-flight check (forward + backward +
>   gradient sanity on a hand-built example). Run before any real training run.
> - **`lora_utils.py`** — shared LoRA wrap (`apply_lora`), trainable-only
>   checkpointing (`get_trainable_state_dict`), and the sparse new-token patch
>   (`apply_sparse_new_token_patch`): the full embedding/LM-head matrices stay
>   frozen and only the 3 special latent tokens get tiny trainable tables
>   (~15K params instead of ~1.3B).
> - **`apply_qwen35_patch.py`** — the original one-shot script version of the
>   Qwen3.5 cache patch; kept for reference, already applied in `coconut.py`.
>
> ### Extra config keys (on top of upstream's list below)
>
> - **use_lora** (default `true`) — LoRA adapters instead of full finetune.
>   `false` falls back to the full-finetune path.
> - **lora_r / lora_alpha / lora_dropout** (defaults `16 / 32 / 0.05`) —
>   LoRA rank/alpha/dropout. `lora_target_modules` / `lora_modules_to_save`
>   are also read if set; target modules are otherwise auto-detected from the
>   model's linear layers.
> - **force_8bit_optimizer** (default `false`) — use bitsandbytes
>   `PagedAdamW8bit` over LoRA params. Only needed if something big is
>   unfrozen; plain AdamW over the adapters is the default.
> - **clip_grad_norm** (default `1.0`) — gradient clipping max norm; `0`
>   disables. Prevents the unclipped-blowup → NaN failure mode.
> - **save_every_n_steps** (default `250`) — periodic mid-epoch checkpoints
>   (`periodic_checkpoint_epoch{N}_step{M}`); `0` disables. These use a
>   distinct prefix so the epoch-based auto-resume ignores them (resume from
>   one manually by path if a session dies mid-epoch).
> - **eval_max_examples** (default unset = full set) — caps only the
>   expensive `generate()`-based accuracy eval, not the cheap forward-pass
>   eval loss.
> - **eval_every_n_epochs** (default `1`) — run the generation eval only
>   every N epochs (plus always on the final epoch and in `only_eval` mode).
> - **wandb_mode** — e.g. `disabled`; forwarded to `WANDB_MODE` before
>   `wandb.init` so logging never blocks on an interactive login prompt.
> - **disable_cudnn_conv** (default `true`) — disables cuDNN globally as a
>   workaround for cuDNN's engine search failing on T4 (sm_75, no bf16 conv
>   support) once latent tokens change input shapes. Re-enable on Ampere+.
>
> NaN/Inf loss or gradients stop the run immediately with a clearly-labeled
> emergency checkpoint (`nan_checkpoint_*` / `nangrad_checkpoint_*`) instead
> of silently training on garbage.
>
> ### Hardware notes
>
> Developed and run on 2× T4 (14.56 GiB each, Kaggle). The memory stack that
> makes a ~4B model fit there: `device_map="auto"` sharding + bf16 +
> gradient checkpointing (`use_cache=False`, see the no-cache fix above) +
> LoRA (~32.5M trainable of ~4.24B total, ~0.77%) + `PYTORCH_CUDA_ALLOC_CONF`
> expandable segments. Reference (non-fused) kernels are used throughout
> because `flash-linear-attention` / `causal_conv1d` wheels are typically
> unavailable there — expect slow steps; that is normal, not a bug.
>
> ---
>
> *Upstream documentation follows.*
>
> ## Getting Started
Clone repo:
```
git clone git@github.com:facebookresearch/coconut.git
cd coconut
```

Setup environment:
```
conda create --name coconut python=3.12
conda activate coconut
pip install -r requirements.txt
```

The code relies on [wandb](https://wandb.ai/site/) for logging. Please log in your wandb account following this [document](https://docs.wandb.ai/ref/cli/wandb-login/) before running any experiments.

## Data

The data for training and evaluation should be presented as a json file like below:

```python
[
  {
    "question": "...",
    "answer": "...",
    "steps": ["...", "...", ...]
  },
  ...
]
```

The file should contain a list of data points. Each data point is composed of a question (str), an answer (str), and a list of steps (str), where each of them is a string.

For example, you can download and process the [GSM8K](https://arxiv.org/abs/2110.14168) dataset (with [augmented training and validation sets](https://github.com/da03/Internalize_CoT_Step_by_Step/tree/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k)) by running:

```bash
bash preprocessing/gsm_icot.bash
```

## Arguments

The configuration of a run should be specified in a yaml file (an example can be found [here](args/gsm_coconut.yaml)).

- **General settings**

  - **project**: Project name for wandb
  - **save_path**: Your path to store the checkpoints
  - **only_eval**: If true, only load a model and test on the data from `val_path` (must used along with `load_model_path`). Otherwise, train the model on `train_path` and test on `val_path` after every epoch.

- **Method**
  - **coconut**: Train coconut model
  - **cot**: Train cot model
  - **no_thoughts**: Train coconut (w/o thought) model
  - **no_cot**: Train no-cot model

- **Training settings**

  - **c_thought**: Number of continuous thoughts for each reasoning step
  - **epochs_per_stage**: Number of epochs for every training stage
  - **max_latent_stage**: The maximum number of training stages (in addition to the initial stage)
  - **pad_latent_to_max**: If the number of reasoning steps is fewer than the index of current training stage, pad the number of continuous thoughts.
  - **save_only_improve**: Save the model only when there the best validation accuracy is updated. Recommended to set `False` for Coconut model training, because otherwise the checkpoints in the last stage might now get saved.
  - **uniform_prob**: The probability to mix data from other stages. 0 for standard experiment, 0.3 for analysis experiment.
  - **model_id**: Huggingface model id to load as the initialization, e.g., `openai-community/gpt2`
  - **load_model_path**: The path to a checkpoint to load. Used in two cases: (1) for evaluation (2) to initialize coconut from a CoT-tuned model.
  - **seed**: Random seed.
  - **resume**: The epoch to resume. Can be used when we want to skip the initial training stages.
  - **bf16**: Whether to use bf16 training.
  - **train_path**: Path to the training set.
  - **val_path**: Path to the validation or test set (depending on `only_eval`)
  - **reset_optimizer**: Whether to reset the optimizer when swtiching training stages.
  - **batch_size_training**: Batch size to train the model per GPU.
  - **debug**: If true, there is no wandb and model saving. A subset of data will be used.
  - **gradient_accumulation_steps**: Gradient accumulation steps
  - **num_epochs**: Maximum training epoches.
  - **lr**: Learning rate
  - **weight_decay**: Weight decay


## Training

Run the following commands (replacing `N_GPUS` and `PATH_TO_ARGS`):

```
torchrun --nnodes 1 --nproc_per_node N_GPUS run.py PATH_TO_ARGS
```

## Reproducing Experiments

Here we provide instructions to reproduce our experiments in the paper.

All the commands below assume 4 * A100 (80GB) GPUs. You may change the corresponding arguments in the config file (`batch_size_training`, `gradient_accumulation_steps`) and `nproc_per_node` when launching the run, to adapt your resources.


### GSM8K

Preprocessing data:

```bash
bash preprocessing/gsm_icot.bash
```

First train the model with CoT (as the stage 0 training)

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_cot.yaml
```

Select a checkpoint as the initialization of Coconut (the validation accuracy is expected to be around 40%). Replace the `load_model_path` in the [args/gsm_coconut.yaml](args/gsm_coconut.yaml) with your selected checkpoint, and run:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut.yaml
```

Find the checkpoint with best validation accuracy, and put the path as `load_model_path` in [args/gsm_coconut_eval.yaml](args/gsm_coconut_eval.yaml). To evaluate:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut_eval.yaml
```

### ProntoQA

Please clone the official [github repo](https://github.com/asaparov/prontoqa/tree/f0145b867b3c106285ec9ea1941a3f6eb7c6162d) of [ProntoQA](https://arxiv.org/pdf/2210.01240) and generate a raw dataset with:

```bash
cd prontoqa
python run_experiment.py --model-name json --model-size dummy --ordering random --num-trials 10000 --few-shot-examples 0 --ontology fictional --min-hops 5 --max-hops 5 --hops-skip 1
```

Then copy the generated `5hop_0shot_random.json` file to `data` directory, and preprocess the dataset with:

```bash
python preprocessing/prontoqa.py
```


Then run the following to train the model:
```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/prontoqa_coconut.yaml
```

Find the checkpoint with best validation accuracy, and put the path as `load_model_path` in [args/prosqa_coconut_eval.yaml](args/prosqa_coconut_eval.yaml). To evaluate:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/prosqa_coconut_eval.yaml
```


### ProsQA

The ProsQA dataset is at [data/prosqa_*.json](data).

Then run the following to train the model:
```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/prosqa_coconut.yaml
```

Find the checkpoint with best validation accuracy, and put the path as `load_model_path` in [args/prosqa_coconut_eval.yaml](args/prosqa_coconut_eval.yaml). To evaluate:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/prosqa_coconut_eval.yaml
```




## Citation
If you use this code base in your research, please cite our paper with the following BibTex entry:
```bibtex
@article{hao2024training,
  title={Training Large Language Models to Reason in a Continuous Latent Space},
  author={Hao, Shibo and Sukhbaatar, Sainbayar and Su, DiJia and Li, Xian and Hu, Zhiting and Weston, Jason and Tian, Yuandong},
  journal={arXiv preprint arXiv:2412.06769},
  year={2024}
}
```

## License
This code is released under the MIT license (see [LICENSE](LICENSE)).