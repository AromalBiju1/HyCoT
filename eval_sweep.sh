#!/usr/bin/env bash
# Eval-only sweep over saved checkpoints. No training.
# Usage: bash eval_sweep.sh <config.yaml> <ckpt_dir> <n_examples> <epoch> [<epoch> ...]
# Example: bash eval_sweep.sh args/gsm_coconut.yaml /kaggle/working/checkpoints_r16/<run_name> 200 4 5 6
# Ablation: ABLATION=zero|embed|noise|shuffle bash eval_sweep.sh ...   (default none; Coconut configs only)
#
# Stage mapping: run_single_gpu.py picks the stage as (resume // epochs_per_stage)
# when it starts the loop at epoch=resume. Checkpoint N was saved AFTER epoch index
# N-1, so resume must be N-1 to evaluate it at the stage it was trained on.
# (For no_cot / cot configs the stage is forced to 0, so this is harmless.)
set -euo pipefail

CFG="$1"; CKDIR="$2"; N="$3"; shift 3
mkdir -p eval_logs
ABL="${ABLATION:-none}"
TAG="$(basename "$CKDIR")_abl-${ABL}"

for E in "$@"; do
  LOG="eval_logs/${TAG}_e${E}_n${N}.log"
  echo "=== ${TAG} ablation=${ABL} checkpoint_${E}  n=${N}  (resume=$((E-1))) ==="
  python run_single_gpu.py "$CFG" \
    --only_eval true \
    --load_model_path "${CKDIR}/checkpoint_${E}" \
    --resume $((E-1)) \
    --eval_max_examples "$N" \
    --latent_ablation "$ABL" \
    --seed 0 2>&1 | tee "$LOG" | grep -aE "Accuracy on validation|CoT match|latent" | tail -3
done

echo; echo "Summary:"; grep -aH "Accuracy on validation" eval_logs/${TAG}_e*_n${N}.log
