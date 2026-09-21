# Do continuous thoughts help Qwen3.5-4B (hybrid Gated DeltaNet)? A small-data check

**Short answer:** in this setup, no. Coconut matched a no-thought control, and the
model did not use the *content* of its latent thoughts. This is a result about a
500-example LoRA regime, not about the method or the architecture in general.

## Setup

| Item | Value |
|---|---|
| Model | `huihui-ai/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated` (24 linear-attn / 8 full-attn layers) |
| Training | LoRA r=16, alpha=32, lr 1e-4, bf16, batch 1, 10 epochs, seed 0 |
| Data | `gsm_train_500.json` (500 GSM8K examples) |
| Curriculum | c_thought=2, 3 epochs per stage; e1–3 stage 0 (0 latents), e4–6 stage 1 (2 latents), e7–9 stage 2 (4 latents), e10 stage 3 (6 latents) — `stage = epoch // 3` (0-indexed) / `stage = (epoch_1idx-1) // 3` |
| Eval | greedy generation on the first 200 of 500 validation problems (same 200 for every run) |
| Hardware | Kaggle T4 x2, reference DeltaNet kernels (no fla / causal-conv1d) |

Three training runs share this recipe:

- **Coconut**: latent tokens fed back as hidden states (the method).
- **No-thought**: same curriculum, same CoT truncation, no latent tokens (the control the paper calls "w/o thought").
- **No-CoT**: answer-only targets.

## Results (n=200, correct / accuracy, 95% Wilson CI about +-7 points)

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

Latent ablations are inference-only, applied to the trained Coconut checkpoint:
`embed` feeds the raw `<|latent|>` token embedding instead of the fed-back hidden
state; `shuffle` feeds the thought computed for the previous question.

## Findings

1. **CoT text carries the accuracy.** No-CoT 0.22 vs about 0.55 for CoT variants at stage 0 (p < 0.001).
2. **Coconut equals the no-thought control.** Epoch 6: 99 vs 97 of 200 (p = 0.92). Epoch 4: 91 vs 80 (p = 0.31), not significant.
3. **The model does not read latent content.** At epoch 6, removing it (98) or shuffling it (89) is not distinguishable from normal latents (99); at epoch 4, shuffling gave exactly the same score (91).
4. **Replacing a CoT step with latents costs accuracy and does not recover it.** Stage 0 about 0.55, stage 1 about 0.40-0.50, for Coconut and no-thought alike.
5. **Checkpoint-to-checkpoint variance is large.** The no-thought run moved 0.40 to 0.485 between epoch 4 and 6 (p = 0.11), so gaps below about 10 points are not interpretable here. An apparent `embed` advantage at epoch 4 (0.52 vs 0.40 no-thought, p = 0.02) did not replicate at epoch 6 and should be treated as noise.

## Why this does not test the method

- The Coconut paper (arXiv:2412.06769) trains GPT-2 with full fine-tuning on 385,620 synthetic GSM8K training examples (Table 3 in Appendix A.3) until epoch 50 (Section 5.1: "For all datasets, after the standard schedule, the model stays in the final training stage, until reaching 50 epochs") — verified against the PDF. Its GSM8k Table 1 numbers (verified against HTML/PDF): CoT 42.9±0.2, Coconut 34.1±1.5, w/o thought 21.6±0.5, pause as thought 24.1±0.7, no-CoT 16.5±0.5. Here: 500 examples, LoRA r16, 10 epochs.
- Every run memorizes its training set (train loss near 0 within a few epochs) and validation loss is lowest around epoch 1-3. Latent stages start at epoch 4, after memorization has begun.
- Epochs 7-10 (stages 2 and 3) are heavily overfit and were not used for conclusions.
- n=200 with one seed. No paired per-question analysis (only totals were logged).
- An r64 run was worse than r16 (0.36 at stage 0 / 0.22 at stage 1 on n=50, from `coconut_runs_condensed_log.txt` / `coconut_metrics.csv`: r64 18/50 at epoch 3, 11/50 at epoch 6) and was stopped early.

## What would make this a real test

1. Thousands of training examples, about 3 epochs per stage, checkpoint by validation loss.
2. A pure-attention model of similar size as a control, to separate hybrid-architecture effects from the small-data regime.
3. Compare the scale of fed-back hidden states with normal token embeddings.
4. Full 500-problem validation set, several seeds, per-question logging for paired (McNemar) tests.

## Reproduce (eval only)

```bash
# checkpoint N is evaluated with resume=N-1 so the stage matches
bash eval_sweep.sh args/gsm_coconut_r16.yaml <ckpt_dir> 200 4 6
ABLATION=embed bash eval_sweep.sh args/gsm_coconut_r16.yaml <ckpt_dir> 200 4 6
ABLATION=shuffle bash eval_sweep.sh args/gsm_coconut_r16.yaml <ckpt_dir> 200 4 6
bash eval_sweep.sh args/gsm_nothoughts_r16.yaml <nothought_ckpt_dir> 200 4 6
bash eval_sweep.sh args/gsm_nocot_baseline.yaml <nocot_ckpt_dir> 200 3
```

Modes: `none | zero | embed | noise | shuffle` (eval-only; the code refuses to run them in training).
Data: `coconut_eval_n200.csv`. Raw logs: `eval_logs.zip` — not committed to git; attached to the GitHub Release for this branch (or Kaggle dataset if linked in the Release notes).
