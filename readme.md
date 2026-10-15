# AE-PSL

**AutoEncoder-Compressed Parallel Split Learning for Pre-Trained Model Fine-Tuning**

Distributed Fine-Tuning (DFT) of large-scale Foundation Models (FMs) on resource-constrained edge devices is limited by local compute constraints and communication overhead. Parallel Split Learning (PSL) reduces client-side computation by keeping few model layers on each client and offloading the remaining computation to the server; however, clients must exchange intermediate activations and gradients with the server at every training step. Existing SL communication-compression methods mainly rely on task-agnostic heuristics, such as sparsification and quantization. While learnable SL compressors can better adapt to intermediate representations, they require co-training with the target model. Therefore, directly inserting them into off-the-shelf FMs introduces feature-distribution misalignment and degrades DFT performance. To address this, we propose AE-PSL, a communication-efficient PSL framework that compresses intermediate activations and gradients using a lightweight AutoEncoder (AE) placed at the split layer. To ensure compatibility of AE compression with pre-trained FMs, AE-PSL introduces a novel two-stage alignment mechanism, which adapts the AE to the pre-trained model’s feature manifold and client-specific feature distributions before DFT. We evaluate AE-PSL across four vision datasets and various communication reduction settings, showing that it preserves downstream accuracy at 10.2× reduction; at the same communication budget, the strongest evaluated heuristic baseline trails AE-PSL by 5.2 percent point. This demonstrates that AE-PSL offers a stronger accuracy–communication trade-off than heuristic SL compression.

[Paper](https://arxiv.org/abs/2607.17913) · [Project](https://nousphera.github.io/AE_PSL/) · [Citation](CITATION.cff)

## Setup

Use Python 3.12 or 3.13. The tested stack is Python 3.12, PyTorch 2.8 and torchvision 0.23.

```bash
git clone https://github.com/Nousphera/AE_PSL.git
cd AE_PSL
bash scripts/setup_env.sh .venv
source .venv/bin/activate
python -m pytest -q
ae-psl --help
```

`PYTHON=python3.13 bash scripts/setup_env.sh .venv` selects another supported interpreter. The script creates a new environment, installs the package with tested constraints, and checks dependencies. Choose a fresh path if an environment already exists. For CUDA, install the matching PyTorch 2.8/torchvision 0.23 wheels for your hardware in that environment.

## Datasets

The preparation CLI writes decoded images, SHA-256 checksums, class mappings and explicit split membership. The trainers consume these manifests directly. Select the datasets your experiment needs:

```bash
python scripts/prepare_data.py images --dataset cifar100 --download
python scripts/prepare_data.py images --dataset food101 --download
# Supply an already acquired official SUN397 archive; its published URL is currently unavailable.
python scripts/prepare_data.py images --dataset sun397 --archive /path/to/SUN397.tar.gz
# Supply licensed ImageNet images in train/<class>/ and test/<class>/ folders.
python scripts/prepare_data.py images --dataset imagenet100 --source /path/to/imagenet100
# Import LEAF's writer-preserving train/*.json and test/*.json splits.
python scripts/prepare_data.py femnist --source /path/to/leaf/femnist/data
```

[Dataset instructions](docs/DATA.md) cover official acquisition, pinned Hugging Face FEMNIST, folder imports, validation, storage and client partitions. SUN397 uses the inherited 80/20 random-split recipe, not official SUN partitions. The historical ImageNet100 class list is unavailable; record your selected classes when reporting results.

## Run

Check the installed training stack without downloading datasets or checkpoints:

```bash
ae-psl-smoke --seed 42 --output results/smoke.json
```

This runs a real small ViT through general alignment (GA), client-specific alignment (CSA), and distributed fine-tuning (DFT), then saves metrics and model tensors. Synthetic accuracy is not a benchmark result. Existing smoke outputs are preserved; choose another output name to rerun.

After preparing CIFAR-100 and ImageNet100, run a single experiment:

```bash
ae-psl --dataset cifar100 --ae_pretrain_dataset imagenet \
  --torch_data_dir data/datasets --model vit_b_32 --nr_of_clients 5 \
  --batch_size 125 --nr_of_epochs 10 --split_layer 5 --ae_latent_dim 64 \
  --random_seed 42 --save_final_model true --experiment_name cifar100
```

Replace `--dataset cifar100` with `food101` or `sun397` for global evaluation, or `femnist` for personalized writer evaluation. FEMNIST defaults to 10% of each writer's samples; `--femnist_fraction 1` retains all samples. GA defaults to 30 epochs on the supplied ImageNet100 subset, followed by one CSA epoch and 10 DFT epochs. The first run downloads torchvision's pretrained ViT weights. Full experiments require substantial compute and storage.

For the downstream-GA ablation, set `--ae_pretrain_dataset` to the same dataset as `--dataset`; only downstream training examples enter GA optimization. `--val_mode early_saving --val_split 0.1` enables held-out checkpoint selection. The primary campaign command saves per-job metrics and completion state and supports `--resume_from_manifest PATH`. The `run_*.sh` scripts provide larger sweeps.

[Reproducibility](REPRODUCIBILITY.md) explains the serialized single-process protocol, explicit logit-gradient communication correction, evaluation conventions, validated scope and benchmark limitations. [Provenance](PROVENANCE.md) records source history. Code uses the existing [MIT license](LICENSE).
