# Dataset preparation

Run commands from the repository root with the setup environment active. `ae-psl-data` and `python scripts/prepare_data.py` expose the same CLI. `--root PATH` changes the preparation root; pass the same path to `ae-psl --torch_data_dir PATH`.

## Public image datasets

```bash
python scripts/prepare_data.py images --dataset cifar100 --download
python scripts/prepare_data.py images --dataset food101 --download
python scripts/prepare_data.py images --dataset sun397 --download
```

CIFAR-100 uses the [Toronto dataset](https://www.cs.toronto.edu/~kriz/cifar.html) and Food101 uses the [ETH release](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/). Their official archive URLs responded to bounded requests on 2026-10-15. The SUN397 URL published in [torchvision's implementation](https://github.com/pytorch/vision/blob/v0.23.0/torchvision/datasets/sun397.py) returned 404 that day. `--download` still targets that authentic endpoint and fails clearly if unavailable; no substitute corpus is silently selected. Import a previously acquired official archive instead:

```bash
python scripts/prepare_data.py images --dataset sun397 --archive /path/to/SUN397.tar.gz
```

`--archive` also works for the original CIFAR-100 and Food101 archives. Acquisition checks the pinned torchvision release's published MD5, records SHA-256, and extracts only regular files and directories beneath a staging root. Absolute paths, traversal, symlinks, hard links and special archive entries are rejected. CIFAR-100's Python archive is decoded only after the official checksum passes.

CIFAR-100 and Food101 preserve provider train/test splits and class ordering. SUN397 preserves the source's 80/20 recipe with torch seed 42; paths are sorted lexically before permutation to remove filesystem-order dependence. This is neither the official ten-partition SUN protocol nor a recovered historical image manifest. The chosen membership is recorded in `manifest.json`.

Downloads and extraction can require tens of GB for large corpora. During preparation the archive, extracted originals and decoded PNG outputs coexist. Ensure adequate storage before starting. No full corpus is bundled or downloaded by tests.

## Folder import and ImageNet100

All four image datasets can be imported from explicit class folders:

```text
source/
  train/class_a/image1.jpg
  train/class_b/image2.jpg
  test/class_a/image3.jpg
  test/class_b/image4.jpg
```

```bash
python scripts/prepare_data.py images --dataset imagenet100 --source /path/to/source
# The same --source interface accepts cifar100, food101 and sun397.
```

Both splits must contain the same class folders. Classes are ordered lexically; each image is decoded, converted to RGB, and stored as a PNG. The manifest retains the original relative path, numeric label and prepared-image checksum. Symlink escapes and hard-linked duplicate files, including train/test overlap, are rejected.

Expected class counts are 100/101/397/100 for CIFAR-100/Food101/SUN397/ImageNet100. `--allow-subset` explicitly marks smaller generated fixtures; it does not turn them into full benchmark datasets. Imported CIFAR-100 uses the original raw 0–255 tensor contract; both GA feature extraction and DFT apply the required ViT resize and normalization.

Obtain ImageNet images under the [provider's terms](https://www.image-net.org/download.php). Use training images under `train/` and your held-out evaluation images under `test/`. This tool neither grants access nor downloads ImageNet. The historical 100-class list and exact image membership were not recovered. The prepared manifest records your actual class list and explicitly marks that limitation. The trainer's dataset name is `imagenet`, while preparation uses `imagenet100`.

## FEMNIST

The primary personalized trainer requires a prepared writer manifest. Choose one source:

```bash
# LEAF JSON: no optional Hugging Face dependency required.
python scripts/prepare_data.py femnist --source /path/to/leaf/femnist/data

# Alternatively, download a pinned provider revision.
python -m pip install -e '.[femnist]' -c constraints-tested.txt
python scripts/prepare_data.py femnist --revision REPLACE_WITH_40_CHARACTER_COMMIT
```

For the Hugging Face command, copy the full commit SHA from the [flwrlabs/femnist revision history](https://huggingface.co/datasets/flwrlabs/femnist/commits/main) and replace the placeholder. Only a full SHA is accepted. The provider's single `train` split contains pooled writer examples, not an evaluation split. The preparation records its revision and fingerprint and preserves `writer_id` and 62-class `character` labels. The optional provider integration is covered with bounded interface fixtures; the full remote corpus has not been downloaded in this release.

For [LEAF](https://github.com/TalwalkarLab/leaf/tree/master/data/femnist), supply generated `train/*.json` and `test/*.json` files with `user_data[writer] = {"x": [...], "y": [...]}`. Each image must have 784 finite values in [0,1], and labels must be integers 0–61. Pixels are quantized to 8-bit PNG using round-to-nearest; retain the original JSON files for provenance. Select a LEAF split by samples so each writer has train and test data. A split by writers, missing split, or duplicate writer within one split is rejected rather than silently discarded.

The trainer sorts writer IDs, permutes them with the requested seed, and divides all writers across clients with `array_split`; remainder writers are retained. For pooled HF data, the default 10% per-writer subset has at least two samples, then an 80/20 train/test split with at least one sample each. For explicit LEAF data, subsampling preserves supplied membership with at least one example per split. Use `--femnist_fraction 1` for all data. Every client needs at least one writer, and pooled writers need at least two samples.

Validation is taken only from each client's training data. The partition JSON records every selected sample index and writer assignment; its filename binds source-manifest SHA-256, client count, seed, sample fraction and validation fraction. A changed source or validation configuration creates a distinct cache. With downstream FEMNIST GA, all selected clients' training data are supplied to alignment and their validation examples remain excluded.

## Outputs and validation

```text
data/datasets/<dataset>/
  manifest.json
  images/000000000.png
  ...
  partitions/<signature>.json   # FEMNIST, created by the trainer
```

```bash
python scripts/prepare_data.py validate --manifest data/datasets/cifar100/manifest.json
python scripts/prepare_data.py validate --manifest data/datasets/femnist/manifest.json
```

Preparation validates decoded images and hashes before publishing its output directory atomically. Existing datasets are never overwritten; use a new root for a new preparation. Interrupted staging directories are cleaned by normal Python exception handling. Keep each prepared directory immutable and revalidate before training after copying or moving data. Runtime loaders check structure and file identity; explicit `validate` performs the full image/hash scan. GA caches bind the manifest digest and the supplied FEMNIST partition identity. Changing image bytes without updating the manifest is invalid; validation rejects it.

Tests generate small image and LEAF fixtures, validate/import them, and execute actual GA → CSA → DFT through the primary orchestrator for all five datasets. Only pretrained checkpoint acquisition and image size are replaced by a small random ViT. Official network downloads and GPU benchmark tables remain unrun.
