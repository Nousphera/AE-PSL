#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
common=(--dataset cifar100 food101 sun397 femnist
        --nr_of_clients 5 25 --batch_size 125 --nr_of_epochs 10
        --split_layer 5 --dirichlet_alpha 0.5 --random_seed 42 43 44 --device cuda)
