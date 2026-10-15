#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
python scripts/run_experiment.py "${common[@]}" --method ae-psl \
  --dataset food101 --nr_of_clients 1 --ae_latent_dim 192 96 48 32 24 16 \
  --ae_force_retrain true --experiment_name architecture \
  --ae_type '1_layer_CONV' '1_layer_MLP_(d_d_z)' '2_layer_MLP_(d_d_4_d_z)' \
            '2_layer_MLP_(d_d_d_z)' '3_layer_MLP_(d_d_2_d_4_d_z)' "$@"
