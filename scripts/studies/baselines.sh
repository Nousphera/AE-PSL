#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
python scripts/run_experiment.py "${common[@]}" --method uncompressed "$@"
python scripts/run_experiment.py "${common[@]}" --method rand-top-k \
  --rand_top_k_sparsity 0.15 0.065 0.025 "$@"
python scripts/run_experiment.py "${common[@]}" --method c3-sl \
  --c3_sl_compression_ratio 5 12 34 "$@"
python scripts/run_experiment.py "${common[@]}" --method adc \
  --adc_compression_ratio 0.2 0.1 0.05 "$@"
