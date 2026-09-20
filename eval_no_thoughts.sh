#!/bin/bash
# Evaluate no-CoT baseline on r64 checkpoints (epoch 1 and 2, stage 0)
# Run this on Kaggle

set -e

CKPT_DIR=/kaggle/working/checkpoints_r64

for EPOCH in 1 2; do
  echo "=== Evaluating epoch $EPOCH (no_thoughts baseline) ==="
  python run_single_gpu.py args/gsm_coconut.yaml \
    --no_thoughts true \
    --coconut false \
    --cot false \
    --load_model_path "$CKPT_DIR/checkpoint_$EPOCH" \
    --resume $EPOCH \
    --num_epochs $((EPOCH + 1)) \
    --only_eval true \
    --eval_max_examples 50 \
    --save_path "/kaggle/working/eval_no_thoughts_e${EPOCH}" \
    2>&1 | tee "/kaggle/working/eval_no_thoughts_e${EPOCH}.log"
  echo ""
done

echo "=== Done. Compare eval accuracy from logs ==="
