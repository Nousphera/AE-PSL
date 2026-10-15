#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
python scripts/run_experiment.py "${common[@]}" --method ae-psl \
  --ae_latent_dim 140 64 24 --experiment_name main "$@"
