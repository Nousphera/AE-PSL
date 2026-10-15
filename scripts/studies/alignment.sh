#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# General alignment, client-specific alignment, frozen compressors.
for recipe in 'false false false ae' 'true false false ga' \
              'true false true ga-frozen' 'true true true ga-csa-frozen'; do
  read -r ga csa frozen name <<< "$recipe"
  python scripts/run_experiment.py "${common[@]}" --method ae-psl \
    --ae_latent_dim 36 --ae_general_alignment "$ga" \
    --client_specific_alignment "$csa" \
    --ae_freeze_encoder_during_finetuning "$frozen" \
    --ae_freeze_decoder_during_finetuning "$frozen" \
    --experiment_name "alignment-$name" "$@"
done
