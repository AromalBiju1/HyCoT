# Latent Reasoning on Hybrid Linear-Attention Architectures

**Does continuous latent thought (Coconut) transfer to hybrid gated-DeltaNet
+ full-attention models, or does it break down when three quarters (24 of 32 layers) already
carry their own recurrent state?**

This project started as a port of Meta's [Coconut](https://arxiv.org/abs/2412.06769)
("Training Large Language Models to Reason in a Continuous Latent Space") onto
Qwen3.5-4B, a **hybrid gated-DeltaNet + full-attention** architecture
(24 linear-attention layers : 8 full-attention layers). It is now a standalone
repo, detached from any upstream fork relationship. The original Coconut
codebase and paper are from Meta; this repo adapts the method to an
architecture family the original work never covered, documents what breaks
when you do that, and reports what happens to task accuracy when it works.

## Why this matters

Coconut's core claim is that a model's hidden state can act as a "continuous
thought" — instead of decoding a reasoning step to discrete tokens and
re-embedding it, you feed the raw hidden state back in as the next input
embedding, for `k` latent steps, before letting the model emit tokens again.
Every prior Coconut result (the original paper and follow-ups) runs this on
plain transformer stacks, where every layer's mechanism for carrying
information across positions is the same: attention.

Qwen3.5's DeltaNet layers already maintain their own compressed recurrent
state across the sequence, by design, independent of what Coconut is doing.
Feeding a Coconut-style continuous thought into a model where 3 out of 4
layers have their own internal notion of "carried state" is untested territory.
It's not obvious whether the two mechanisms compose cleanly, interfere, or one
makes the other redundant. That's the open question this repo investigates.

**Relevance and scope, stated plainly:** GSM8K (grade-school arithmetic word
problems) is the standard Coconut benchmark, chosen here for direct
comparability to the original paper, not because arithmetic reasoning is the
end goal. This repo answers an architecture-level mechanistic question — does
recurrent latent-token feedback work on hybrid linear-attention models — not
a domain-level one. Findings here should be read as evidence about the
mechanism, not as a claim about any specific downstream application.

## Status

Experiments on the 500-example LoRA r16 setup are complete; see Results. Larger-data runs, multiple seeds, and a pure-attention control are blocked on compute (Kaggle 2x T4, 30h weekly GPU quota).

## Architecture-specific fixes

Porting Coconut to Qwen3.5 broke several assumptions baked into the original
code, because that code assumes a standard KV-cache attention stack. Every
fix below is marked `Qwen3.5 patch` in source, in `coconut.py` unless noted.

- **Cache is not sliceable KV.** Original Coconut truncates
  `past_key_values` per latent pass (`k[:, :, :n, :]`) to only feed the model
  what it needs. Qwen3.5's cache holds per-layer `LinearAttentionLayer`
  objects with `recurrent_states` / `conv_states` — a compressed running
  state, not a token-indexed KV tensor — so it cannot be sliced this way.
  Fixed by passing the cache through whole instead of truncating it.
  (`apply_qwen35_patch.py` keeps the original one-shot patch script for
  reference; the fix is applied directly in `coconut.py`.)

- **In-place cache updates break autograd.** Transformers'
  `LinearAttentionLayer` updates its state buffers with `.copy_()` — an
  in-place CUDA-graph-friendly optimization for inference. This silently
  overwrites tensors that autograd had saved for the backward pass, so any
  run using `use_cache=True` together with `.backward()` fails or produces
  wrong gradients. Fixed by monkey-patching the update methods to reassign
  out-of-place, and cloning DeltaNet states between Coconut's re-entrant
  forward passes (`_patch_linear_attention_cache_for_training`,
  `_clone_cache_states`).

- **No KV cache under gradient checkpointing.** Checkpointing forces
  `use_cache=False`, so `past_key_values` comes back `None` on every pass —
  the original multi-pass latent loop raised `IndexError` on the very first
  latent-token training epoch. Fixed with full-prefix recompute per latent
  pass (offset 0, only the new slice's logits are used) whenever no cache is
  present.

- **`fused_recurrent` kernels have no backward pass.** Installing
  `flash-linear-attention` for speed causes HF's kernel dispatch to select
  `fused_recurrent_gated_delta_rule` for single-new-token forward calls
  (which is what every latent pass looks like from the cache's point of
  view, regardless of whether the prefix was recomputed). That kernel has no
  backward implementation upstream (`NotImplementedError`, by design — see
  `fla/ops/gated_delta_rule/fused_recurrent.py`). **Tested and confirmed:**
  installing `fla` still fails here even after the full-prefix-recompute fix
  above, because kernel dispatch is keyed on new-token count per call, not on
  whether the prefix was recomputed. Reference (non-fused) PyTorch kernels
  are used throughout as a result — training is correspondingly slower, but
  correct.

- **Single-process multi-GPU.** `device_map="auto"` model-parallel sharding
  is used instead of DDP; dataset/distributed guards, device placement, and
  checkpoint save/load are adapted accordingly (`dataset.py`,
  `run_single_gpu.py`, `lora_utils.py`).

- **Checkpoint auto-resume does not validate LoRA shape.** `run_single_gpu.py`
  silently resumes from any checkpoint found at `save_path`, regardless of
  whether its LoRA rank matches the current run's config. Loading a rank-16
  checkpoint into a rank-64 model produces a wall of
  `size mismatch for ... lora_A/lora_B` errors instead of a clear message.
  **Known issue, fix planned:** validate saved rank against current config
  before `load_state_dict` and fail with an explicit message
  ("checkpoint was rank 16, current config is rank 64, refusing auto-resume")
  instead of attempting the load blind. Until fixed, always point `save_path`
  at a fresh directory when changing any LoRA hyperparameter.

## Usage

### Scripts

- **`run.py` vs `run_single_gpu.py`** — `run.py` is the original Coconut
  entry point: full-parameter finetune under `torchrun` with NCCL/FSDP/DDP
  across datacenter GPUs. `run_single_gpu.py` is this repo's rewrite for
  Kaggle (2× T4): single process, `device_map="auto"` model-parallel
  sharding instead of DDP, LoRA adapters instead of full finetune, plus the
  Qwen3.5 fixes above. Use `run_single_gpu.py` for all runs here; `run.py`
  is kept for reference only and will OOM on this hardware. Note the two
  scripts also differ in auto-resume behavior — see the checkpoint
  auto-resume caveat above before pointing either at an old save dir.
- **`run_single_gpu.py`** — training entry point. Single process,
  `device_map="auto"` sharding, LoRA by default.
  `python run_single_gpu.py args/gsm_coconut.yaml --lr 5e-5 --debug true`
- **`smoke_test_coconut.py`** — fast pre-flight check (forward + backward +
  gradient sanity on a hand-built example). Run before any real training run.
- **`lora_utils.py`** — LoRA wrap (`apply_lora`), trainable-only checkpointing
  (`get_trainable_state_dict`), and the sparse new-token patch
  (`apply_sparse_new_token_patch`): the embedding/LM-head matrices stay
  frozen and only the 3 special latent tokens get tiny trainable tables
  (~15K params instead of ~1.3B).
- **`apply_qwen35_patch.py`** — original one-shot version of the cache patch;
  kept for reference, already applied directly in `coconut.py`.

### Config keys (beyond the standard Coconut set)

| Key | Default | Purpose |
|---|---|---|
| `use_lora` | `true` | LoRA adapters instead of full finetune |
| `lora_r` / `lora_alpha` / `lora_dropout` | `16 / 32 / 0.05` | LoRA hyperparameters |
| `force_8bit_optimizer` | `false` | `PagedAdamW8bit` over LoRA params if needed |
| `clip_grad_norm` | `1.0` | gradient clipping max norm; `0` disables |
| `save_every_n_steps` | `250` | periodic mid-epoch checkpoints; `0` disables |
| `eval_max_examples` | unset (full set) | caps the expensive `generate()`-based accuracy eval only |
| `eval_every_n_epochs` | `1` | run generation eval only every N epochs |
| `wandb_mode` | — | e.g. `disabled`, forwarded before `wandb.init` |
| `disable_cudnn_conv` | `true` | works around cuDNN engine-search failure on T4 (sm_75) |

NaN/Inf loss or gradients stop the run immediately with a clearly-labeled
emergency checkpoint (`nan_checkpoint_*` / `nangrad_checkpoint_*`) instead of
silently continuing to train on garbage.

### Hardware

Developed on 2× T4 (14.56 GiB each, Kaggle). Fitting a ~4B model there:
`device_map="auto"` sharding + bf16 + gradient checkpointing + LoRA
(~0.77–3% trainable) + `PYTORCH_CUDA_ALLOC_CONF` expandable segments.
Reference (non-fused) kernels are used throughout — `flash-linear-attention`
does not currently work here (see fixes above) and `causal_conv1d` is
unavailable on this hardware. Expect slow steps; this is expected, not a bug.

## Experiment design

- **Base model:** `huihui-ai/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated`
- **Task:** GSM8K, 500-example training subset, held-out validation set
- **Comparison axes:**
  - No-CoT, no-thought control (same curriculum and CoT truncation, no latent tokens), and Coconut
  - Inference-time latent ablations (embed, shuffle) on the trained Coconut checkpoint
  - LoRA rank 16 vs 64 (r64 stopped after epoch 6; early small eval, not in the release)
  - Not run: separate full-CoT baseline, pause-token control
- **Metrics:** generation accuracy on a fixed eval subset (first 200 of 500 validation problems, greedy, same 200 for every run; 95% Wilson CI) — held-out eval loss was logged per epoch and is used only for the overfitting caveat below; CoT-match not reported at n=200

## Results

Continuous thought did not outperform the no-thought control in this 500-example LoRA regime. See [docs/RESULTS_latent_ablation.md](docs/RESULTS_latent_ablation.md) for full analysis.

n=200 greedy eval (same 200 problems for every run), 95% Wilson CI ≈ ±7 points. Stage = `(epoch - 1) // 3` for 1-indexed epochs (e1–3 stage 0, e4–6 stage 1 with 2 latents, e7–9 stage 2, e10 stage 3) — matches the log's stage map and the table below.

Base model: `huihui-ai/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated` (abliterated variant), not stock Qwen3.5-4B.

Caveat: validation loss is lowest around epochs 1–3 and every run memorizes its training set (train loss near 0 within a few epochs), so latent stages (epoch 4 onward) begin after memorization has started.

Noise note: differences under about 10 points (about 20 problems) between two runs are not interpretable at n=200 with one seed; the epoch-4 embed-vs-no-thought gap (0.52 vs 0.40) did not replicate at epoch 6 (0.490 vs 0.485).

| Run | Epoch (stage) | Correct | Acc |
|---|---|---|---|
| No-CoT | 3 (0) | 44 | 0.22 |
| No-thought | 3 (0) | 107 | 0.535 |
| Coconut | 3 (0) | 114 | 0.570 |
| No-thought | 4 (1) | 80 | 0.400 |
| Coconut, normal latents | 4 (1) | 91 | 0.455 |
| Coconut, shuffled latents | 4 (1) | 91 | 0.455 |
| Coconut, latent content removed (`embed`) | 4 (1) | 104 | 0.520 |
| No-thought | 6 (1) | 97 | 0.485 |
| Coconut, normal latents | 6 (1) | 99 | 0.495 |
| Coconut, latent content removed (`embed`) | 6 (1) | 98 | 0.490 |
| Coconut, shuffled latents | 6 (1) | 89 | 0.445 |

Data: `coconut_eval_n200.csv`. Raw logs: `eval_logs.zip` (see Release). The claims above are only reproducible with the patched `coconut.py` + `run_single_gpu.py` + `eval_sweep.sh` in this branch.

## Open questions

- Do latents help at thousands of training examples? Untested.
- Is the null result specific to the hybrid architecture? Needs a pure-attention control at the same budget.
- Stages 2-3 (4 and 6 latents) scored lower in early small checks that are not in the release. Those epochs are heavily overfit and there is no matched no-thought comparison at n=200, so curriculum effect vs latent depth is unresolved.
- An early r64 run scored lower than r16 at matched epochs, was stopped at epoch 6 and overfit. Not in the release; no conclusion about capacity is drawn.

## Data

Training/eval data follows the original Coconut format — a JSON list of
`{"question": ..., "answer": ..., "steps": [...]}` objects. See
`preprocessing/gsm_icot.bash` to regenerate the GSM8K split used here.

## Acknowledgments and license

The core Coconut method, training loop structure, and original codebase are
from Meta's paper below. This repo's contribution is the Qwen3.5 hybrid-
architecture port, the fixes documented above, and the experiments and
findings reported here. Released under the MIT license (see `LICENSE`),
consistent with the upstream project.

```bibtex
@article{hao2024training,
  title={Training Large Language Models to Reason in a Continuous Latent Space},
  author={Hao, Shibo and Sukhbaatar, Sainbayar and Su, DiJia and Li, Xian and Hu, Zhiting and Weston, Jason and Tian, Yuandong},
  journal={arXiv preprint arXiv:2412.06769},
  year={2024}
}
```
