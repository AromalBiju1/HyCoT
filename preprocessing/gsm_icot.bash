#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Create data directory if it doesn't exist
mkdir -p data

# Skip entirely if the final json files already exist. Without this check,
# every rerun of this script (e.g. re-running a notebook cell, or a fresh
# Kaggle session that re-executes the whole notebook top-to-bottom)
# unconditionally re-downloads and re-converts, silently overwriting
# whatever's in data/gsm_train.json/gsm_valid.json/gsm_test.json --
# including a hand-subsetted file built on purpose for a fast test run.
# There was no warning when this happened before; it just quietly replaced
# a 500-example subset with the full converted dataset.
already_have_all=true
for split in train valid test; do
  if [ ! -f "data/gsm_${split}.json" ]; then
    already_have_all=false
  fi
done

if [ "$already_have_all" = true ] && [ "$1" != "--force" ]; then
  echo "data/gsm_{train,valid,test}.json already exist -- skipping download+convert."
  echo "Pass --force as the first argument if you actually want to regenerate from source."
  exit 0
fi

if [ "$1" = "--force" ]; then
  echo "Forcing regeneration -- existing json files (if any) will be overwritten."
fi

# Download and process GSM8K dataset for Internalize CoT
wget https://media.githubusercontent.com/media/da03/Internalize_CoT_Step_by_Step/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k/train.txt -O data/gsm_train.txt
wget https://raw.githubusercontent.com/da03/Internalize_CoT_Step_by_Step/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k/valid.txt -O data/gsm_valid.txt
wget https://raw.githubusercontent.com/da03/Internalize_CoT_Step_by_Step/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k/test.txt -O data/gsm_test.txt

for split in train valid test; do
  python preprocessing/gsm_icot.py ${split}
  rm data/gsm_${split}.txt
done