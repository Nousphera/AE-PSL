# AE-PSL

**AutoEncoder-Compressed Parallel Split Learning for Pre-Trained Model Fine-Tuning**

Bas Meuwissen · Vasileios Tsouvalas · Nirvana Meratnia

[Paper](https://arxiv.org/abs/2607.17913) · [Project](https://nousphera.github.io/AE_PSL/) · [Reproducibility](REPRODUCIBILITY.md) · [Provenance](PROVENANCE.md)

AE-PSL compresses patch-token activations and their gradients at a ViT split point. General alignment (GA) learns an autoencoder on public features; client-specific alignment (CSA) adapts local encoders and averages decoders; distributed fine-tuning (DFT) freezes the autoencoders while training LoRA adapters and the classifier. CLS tokens travel uncompressed.

This release contains the original experiment implementation, corrected training paths, and a bounded CPU integration run. It uses serialized boundaries in one process. Detached client logits return explicit logit gradients for the server backward pass; those extra bytes are included. Scalar losses alone cannot provide those gradients. This correction and the distinction from the paper's communication claims are detailed in [Reproducibility](REPRODUCIBILITY.md#transport-and-claims).

## Install and run

Use Python 3.12–3.13 in an isolated environment. The tested CPU environment is Python 3.12, PyTorch 2.8 and torchvision 0.23.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints-tested.txt '.[test]'
python -m pytest -q
ae-psl-smoke --seed 42 --output results/smoke.json
```

The smoke run downloads nothing. It trains a real, randomly initialized two-block ViT through GA → CSA → DFT on small synthetic image datasets. It verifies frozen weights and gradient flow, then evaluates both global (AE bypassed) and local (AE active) models. The JSON records configuration, dependency versions, metrics, communication, and a model-state digest. A matching NPZ contains all client/server model tensors and loads with `numpy.load(path, allow_pickle=False)`. Existing outputs are refused; checkpoints are for inspection/inference, not optimizer-resume training. Synthetic accuracy is **not a paper benchmark**.

## Paper experiments

```bash
# Supply ImageNet100 first; see the required layout and limitations below.
ae-psl --dataset cifar100 --model vit_b_32 --nr_of_clients 5 \
  --batch_size 125 --nr_of_epochs 10 --split_layer 5 \
  --ae_latent_dim 24 64 140 --random_seed 42 43 44 \
  --save_final_model true --experiment_name cifar100_three_seeds
```

This is a large experiment sweep, with 30 GA epochs and one CSA epoch by default. Torchvision downloads the named pretrained ViT weights and CIFAR-100 when needed. [Dataset preparation, FEMNIST installation, configuration, and evaluation details](REPRODUCIBILITY.md#datasets-and-experiments) include the fixed SUN397 split and ImageNet100 class-list limitation. `ae-psl --help` lists the full configuration; existing `run_*.sh` scripts use the installed module. Optional audio research files are retained for provenance and are outside this vision release's supported scope.

Code is distributed under the repository's existing [MIT license](LICENSE). Please cite [CITATION.cff](CITATION.cff).
