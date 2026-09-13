# Getting Coconut (Continuous Latent Thought) Running on Qwen3.5's Hybrid DeltaNet Architecture, Under LoRA

## Summary

[Coconut](https://arxiv.org/abs/2412.06769) (Hao et al., Meta FAIR, Dec 2024) trains a
language model to reason in continuous latent space instead of emitting explicit
chain-of-thought tokens, by feeding the model's own last hidden state back in as the
next input embedding. The original implementation was built and validated only against
plain transformer architectures (GPT-2, Llama) with full-parameter finetuning.

This project ports Coconut's training mechanism to **Qwen3.5-4B**, whose backbone is a
**hybrid architecture**: 75% of layers use Gated DeltaNet (linear attention with a
recurrent state), 25% use standard full attention, in a 3:1 ratio. Nobody had previously
tried pushing Coconut's cross-pass gradient flow through a Gated DeltaNet layer's
recurrent state. Separately, full-parameter finetuning of a 4B model turned out to be
impractical on the available compute (2x Kaggle T4, 16GB each), so the project pivoted
to LoRA plus a small custom bridge for the new special tokens Coconut requires.

Getting this combination to actually run correctly surfaced a long chain of real bugs —
in HF's `transformers` cache handling, in `flash-linear-attention`'s kernel coverage, in
`accelerate`'s device-sharding behavior, in cuDNN's algorithm selection on Turing
hardware, and in the original Coconut reference implementation's own cache-handling
logic once no-cache training was introduced. Each of these had to be independently
diagnosed from a real traceback, not guessed at. This document is a record of that
process, plus the actual experimental result reached so far.

**This is not a claim that Coconut "works" or "doesn't work" on this architecture.**
It's a record of what it took to get a fair test running at all, and an honest,
partial, in-progress readout of what that test shows so far.

---

## Why this combination is harder than the original paper's setup

- **Architecture**: Coconut assumes the model's last hidden state can be fed straight
  back in as a next-step input embedding. This is architecture-agnostic in principle,
  but nobody had verified it against a model with a *recurrent* linear-attention state
  (Gated DeltaNet) that persists and mutates across the forward pass, as opposed to a
  pure attention KV cache.
- **Compute budget**: full-parameter finetuning of a 4B model doesn't fit in 2x16GB T4s
  without aggressive tricks (paged 8-bit optimizers, model-parallel sharding, gradient
  checkpointing) that create their own compounding failure modes. LoRA was adopted
  specifically to make the memory budget survivable — this is a materially different
  training regime than the paper's, not just a smaller version of the same thing.
- **Reference kernels only**: `flash-linear-attention`'s `fused_recurrent` kernel — the
  one Coconut's per-latent-pass re-entry actually triggers — has **no backward pass
  implemented at all**, by explicit upstream design. This isn't a missing pip install;
  it's a real gap in that library for this exact usage pattern. Training had to run on
  HF's plain PyTorch reference kernels for DeltaNet throughout, which are correct but
  meaningfully slower.

---

## Chronological log of real bugs found and fixed

Each of these was found by reading an actual traceback and understanding the root
cause — not by trial-and-error retries.

### 1. In-place cache mutation broke autograd (full-finetune phase)
HF's `cache_utils.py` `LinearAttentionLayer.update_recurrent_state` /
`update_conv_state` perform `.copy_()` **in place** into a persistent buffer (a
CUDA-graph optimization comment marks this as inference-only). This silently mutated
the exact tensor `fla`'s `ChunkGatedDeltaRuleFunction.forward` had just saved via
`ctx.save_for_backward(...)`, breaking autograd within a single forward call — not
specific to Coconut's cross-pass reuse. Fixed with a monkeypatch reassigning instead of
copying, applied at import time in `coconut.py`.

### 2. Single-GPU OOM during backward
14.53/14.56 GiB used on a single T4 during backward on a plain MLP layer. Fixed (for
the full-finetune phase) via `device_map="auto"` to shard the model across both T4s.

### 3. `flash-linear-attention`'s `fused_recurrent` kernel has no backward pass
Distinct from bug #1 — a different kernel path (decode/single-token re-entry, which is
exactly what Coconut's latent-pass loop triggers), with **no backward implementation at
all**, by explicit upstream design ("haven't figured out how to compute dg without
materializing the full hidden states"). Resolution: uninstall `flash-linear-attention`
and `fla-core` entirely; HF's plain PyTorch reference kernels (both chunk and recurrent
paths) are fully differentiable ordinary tensor ops with no such limitation. Training
proceeds correctly on these, just slower.

### 4. Full-finetune memory wall was impractical, LoRA switch
Realized the training script was doing a full finetune (optimizer over all
parameters), conflicting with the project's own stated non-goal of full-finetuning a 4B
model on this hardware. Rewrote the training script to use LoRA (via `peft`):
auto-detected all real `nn.Linear` leaf modules across both DeltaNet's projections
(`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj` — none of which match
any published LoRA target-module list, since those were only ever written for plain
attention architectures) and the standard attention/MLP projections, rather than
hardcoding a target list that would have silently missed DeltaNet's layers.

### 5. New special tokens (`<|start-latent|>`, `<|latent|>`, `<|end-latent|>`) under LoRA
Coconut requires three new vocabulary tokens. The first approach —
`lora_modules_to_save: embed_tokens,lm_head` — unfroze the **entire** 248,080-row
embedding and output matrices (≈635M params each) just to let 3 rows move: ~1.27B extra
trainable params, ~10GB of AdamW optimizer state alone, which OOM'd immediately.
Replaced with a custom sparse patch: the full embedding/lm_head stay frozen, and a tiny
`[3, hidden]` trainable table supplies the override for just those 3 token ids on both
the input (embedding) and output (lm_head) side. This cut trainable-params-for-new-
tokens from ~1.27B to a few thousand.

### 6. Device-placement bugs in the sparse patch (`device_map="auto"`)
Two separate bugs, both from the same root cause: the sparse patch's own small tables
(`nn.Embedding`, `nn.Linear`) default to CPU/fp32 on construction, but the base model is
sharded across GPUs by `accelerate`. First fix moved them to the device/dtype captured
at construction time — insufficient, because `accelerate`'s dispatch hooks can finalize
real shard placement *after* construction. Final fix: self-heal device/dtype at every
forward call against the actual tensor that just passed through the frozen base layer,
rather than trusting any one-time snapshot. Confirmed safe to do even after the
optimizer is constructed, since `nn.Module.to()` mutates parameters in place (preserves
the same Python object identity the optimizer's state dict is keyed against).

### 7. Fragile transitive import chain (`torchvision`/`torchaudio` version mismatches)
`coconut.py` imported `GPT2LMHeadModel` from `transformers` purely for one `isinstance`
check never true for this model — but that import transitively pulled in
`transformers.modeling_layers` → object-detection loss code → `torchvision`, which
broke on a torch/torchvision CUDA-version mismatch on the Kaggle image. Separately,
`peft`'s own `__init__` chain imports `BloomPreTrainedModel`, which transitively pulls
in `torchaudio`, which broke the same way. Fixed by (a) replacing the `GPT2LMHeadModel`
isinstance check with a duck-typed class-name string check, removing the need to import
GPT2's modeling code at all, and (b) uninstalling the unused `torchvision`/`torchaudio`
packages entirely so `transformers`' own availability guards skip them cleanly instead
of eagerly importing and crashing.

### 8. cuDNN "unable to find an engine" on Turing hardware
`RuntimeError: GET was unable to find an engine to execute this computation`, thrown
from `F.conv1d` inside DeltaNet's grouped (depthwise-style, `groups=8192`) conv1d layer.
Occurred on the very first batch of the first epoch where the curriculum actually
inserted real `<|latent|>` tokens (all prior epochs were curriculum stage 0, zero
latent tokens — this was the first forward pass to exercise a genuinely new shape/
pattern). Root cause: T4 is Turing (sm_75), which predates native BF16 tensor-core
support (arrived with Ampere/sm_80); cuDNN's algorithm search for this specific op under
bf16 can fail to find *any* valid engine on this hardware. Fixed by disabling cuDNN for
convolutions, forcing PyTorch's plain (non-cuDNN) conv1d path — a real fix, not a
retry-and-hope, at the cost of somewhat slower conv ops (a small overhead on top of an
already-reference-kernel-only training run).

### 9. First-pass / no-cache branch conflict in `coconut.py` itself
The deepest bug, found and fixed directly by the project owner. `run_single_gpu.py`
enables gradient checkpointing, which forces `use_cache=False`, so the model returns
`past_key_values=None` on **every** pass, not just the first. The original Coconut
`forward()` logic branched on `if kv_cache is None` to mean "this is the first latent
pass" — but with caching disabled, `kv_cache` stays `None` on every subsequent pass too,
so pass 1 silently re-entered the "first pass" branch: it forwarded only a 1-token
slice with no prefix, produced a hidden-states tensor of length 1, then indexed it at an
absolute sequence position far beyond its length, raising an `IndexError`. This didn't
surface until the first epoch where curriculum stage 1 actually inserted a real latent
token (all earlier epochs were stage 0, zero latent tokens — the loop that contains
this bug was never entered). Fixed by splitting the branch explicitly on
`kv_cache is not None` (real cached incremental pass) vs. genuinely-no-cache (recompute
the full prefix from position 0 each time, then keep only the newly-relevant slice of
logits so the concatenated output still covers each position exactly once). Same fix
applied to the final pass, which had the identical bug.

---

## Current experimental status (in progress, not final)

- Training: LoRA (rank 16, auto-detected target modules across both DeltaNet and
  attention/MLP projections) + a sparse trainable patch for the 3 new special tokens,
  on Qwen3.5-4B (`huihui-ai/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated`), on a
  500-example GSM8K (Internalize-CoT-Step-by-Step formatted) subset, on Kaggle T4 x2.
- Curriculum: `epochs_per_stage=3`, `max_latent_stage=3`, `c_thought=2` — standard
  Coconut staged curriculum, ramping from 0 explicit-CoT-replacement to full latent
  reasoning over 9 epochs, then continuing at the final stage.
- Gradient clipping (`max_norm=1.0`) added after an earlier full-finetune attempt
  diverged to NaN via exponential loss growth; no NaN/Inf observed since, across LoRA
  training through curriculum stage 1.
- **Checkpoint comparison so far** (same 50 held-out validation examples, same seed):

  | Checkpoint | Curriculum stage | Latent tokens | Accuracy |
  |---|---|---|---|
  | epoch 3 (checkpoint_3) | 0 | 0 (full explicit CoT) | 27/50 = 0.54 |
  | epoch 5 (checkpoint_5) | 1 | 2 | 26/50 = 0.52 |

  Essentially flat, within noise for n=50. At least one qualitative case (a discount-
  percentage GSM8K problem) shows the model correctly using a value inside the 2 latent
  tokens that it never wrote out explicitly (`2000-600=1400` appears with no visible
  computation of the `600`), suggesting the mechanism is doing *something* real rather
  than acting as an inert placeholder — but a flat accuracy result at this stage doesn't
  yet distinguish between two live hypotheses:
  1. Curriculum hasn't progressed far enough yet (still early, stage 1 of 3) — expected
     to resolve by continuing training to stage 3.
  2. LoRA's capacity (rank 16, ~0.77% of total params) may be insufficient for the model
     to develop and exploit whatever internal representation Coconut's mechanism
     requires, independent of curriculum stage — would require a different
     intervention (higher rank, more target modules) to resolve, not just more epochs.

  These two hypotheses are **not distinguishable from the data collected so far**, and
  won't be until training reaches curriculum stage 3 (epoch 9+) and is compared against
  a from-scratch CoT-only baseline trained under the same LoRA/data/step budget — not
  an earlier-stage Coconut checkpoint standing in for one.

## Honest caveats

- Small dataset (500 examples), small eval set (50 examples) — real signal, but with a
  wide noise band. One flipped example is a 2-point swing in accuracy.
- LoRA is a materially different training regime from the original paper's full
  finetune. A null result here cannot be read as "Coconut doesn't work" in general —
  only as "Coconut, under these specific capacity and data constraints, on this
  specific architecture, did or didn't show a measurable effect at this scale."
- Reference kernels only throughout — no `flash-linear-attention`, no `causal_conv1d`.
  Training is correct but slow; wall-clock cost scales with the number of latent tokens
  per step, which grows with curriculum stage.

## Next steps

1. Complete training through curriculum stage 3 (epoch 9+).
2. Train a from-scratch CoT-only baseline under an identical LoRA/data/step budget
   (not an earlier Coconut checkpoint) for a fair comparison.
3. Compare final-stage Coconut accuracy against that baseline on a larger held-out set
   to reduce noise.
4. If a genuine LoRA-capacity ceiling is suspected after that comparison, test a rank
   increase (e.g. r=16 → 32/64) as a separate, explicit variable — not conflated with
   curriculum progress.
