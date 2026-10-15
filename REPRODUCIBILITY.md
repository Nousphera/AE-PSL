# Method and experiment configuration

AE-PSL compresses split-layer activations and gradients with an autoencoder aligned to a pretrained vision model. General alignment (GA), client-specific alignment (CSA), and distributed fine-tuning (DFT) form the training pipeline.

Reference: [AutoEncoder-Compressed Parallel Split Learning for Pre-Trained Model Fine-Tuning](https://arxiv.org/abs/2607.17913). Paths below are relative to `src/ae_psl/`.

## Paper-to-code map

| Component | Implementation | Behavior |
| --- | --- | --- |
| Public-feature GA, §4.1.1 | `ae_trainers/ae_trainer.py`, `centralized_ae_trainer.py` | Train the AE on features from a frozen ViT |
| Local CSA, §4.1.2 | `trainers/implementations/classification/sasl_trainer.py` | Retain client-specific encoders; average decoder parameters uniformly |
| Frozen-AE DFT, §4.2 | Same trainer and split classification models | Freeze AE weights while retaining gradients through their inputs |
| CLS preservation | `models/auto_encoder.py`, split models | Pass the CLS token and its gradient unchanged |
| Single server backward | `transport.py`, main trainer | Exchange detached tensors and weighted logit gradients |
| Global evaluation, §5.2 | `test_epoch_global_evaluation` | Average client model; bypass AE halves; example-weighted metrics |
| Local evaluation, §5.2 | `test_epoch_distributed` | Local encoder and shared decoder; macro average across clients |
| LoRA | `models/lora_layer.py`, `vision_transformer_base.py` | Adapt ViT MLP linear layers and the classifier |

The torchvision ViT configuration uses MLP adapters; attention Q/K/V and `MultiheadAttention.out_proj` are excluded. Rank and alpha default to 16 and 32. The AE uses LayerNorm, GELU and 0.01 encoder dropout during alignment. DFT switches frozen AEs to evaluation mode while retaining input gradients.

Reconstruction MSE averages over the full token sequence. The preserved CLS token contributes zero error, scaling a patch-only mean by `patches / (patches + 1)`.

## Tensor exchange and communication

The simulator runs in one process with serialized boundaries. Clients send latent patch tokens and CLS tokens as float32 bytes with JSON shape/dtype metadata. The server returns detached logits. Each client computes cross entropy and sends `n_client / n_global_batch * dLoss/dLogits`. One server backward pass produces latent/CLS gradients, which return to the original client graphs.

The objective is a sample-count-weighted average of client mean losses, including unequal batches. Paper §4.2 writes a sum of client losses; the weighting used here is explicit in the trainer and its gradient-equivalence tests.

Backpropagation across a detached boundary requires logit gradients in addition to the scalar loss telemetry described in the paper. DFT counts both, including message headers. Its byte totals therefore have a different scope from scalar-only communication ratios. CSA records raw decoder tensor bytes separately. Network framing, encryption, model distribution and GA data movement are outside these counters. Result keys ending `_in_mb` use MiB (`bytes / 2^20`).

`server_batch_mse` is a local diagnostic comparing client activations with reconstructions. It does not affect optimization or protocol counts. A distributed backend must omit this shared-memory diagnostic or account for the additional exchange. Labels stay in client loss calculations, but gradients may reveal them; this simulator provides no cryptographic or differential-privacy protection.

## Datasets and experiment recipe

Prepare data using [docs/DATA.md](docs/DATA.md). CIFAR-100 and Food101 retain provider splits. SUN397 uses lexically ordered paths and a seeded 80/20 split, separate from the official SUN partition protocol. ImageNet100 uses the class folders you supply and records their exact membership. FEMNIST retains writer identities and binds partition caches to the source manifest and validation settings.

| Setting | Default |
| --- | --- |
| Backbone | torchvision ViT-B/32, `IMAGENET1K_V1` weights |
| Split layer | 5 |
| Compressor | Two-layer MLP `(d, d, d_z)` |
| Latent dimensions | 140, 64 or 24; `d_z=d` selects identity compression |
| Alignment | 30 GA epochs on supplied ImageNet100; one CSA epoch |
| Fine-tuning | 10 epochs; Adam, learning rate `1e-4` |
| Global batch / clients | 125 / 5 or 25 |
| Seeds | 42, 43, 44 |

`run_basic_example.sh` defines a sweep; `ae-psl --help` lists all flags. GA and DFT validation are disabled by default, so test accuracy does not select checkpoints. Enable held-out selection with `--val_mode early_saving --val_split 0.1`. For downstream-GA ablations, only the downstream training examples enter alignment.

## Results, checkpoints and resume

Campaigns write a dated manifest, per-job JSON, `.status.json` and optional checkpoints. A final `completion` record identifies the stage, requested epochs, last completed epoch and early-stop state. Resume with `--resume_from_manifest PATH`; incomplete jobs restart from their beginning and may reuse a compatible GA cache. Results are written atomically, and failed jobs produce an unsuccessful campaign exit.

With validation, `.best.pth` stores the best held-out checkpoint and `.latest.pth` stores the latest epoch separately. Final evaluation reloads the best snapshot. `--save_final_model true` saves that evaluated state. Personalized checkpoints contain all `client_models` plus `server_model`; global checkpoints contain `client_model` plus `server_model`. Checkpoints store model state, not optimizer-resume state. Results distinguish `best_validation_epoch` from `last_completed_epoch`.

GA cache schema 4 binds the seed, training settings, model hash/split, dataset root, manifest SHA-256 and FEMNIST partition identity. Keep prepared data immutable and run hash validation after copying or editing it. For unprepared folders, use a fresh cache after changing files. `--ae_specific_weights_path` explicitly selects an AE checkpoint; match its configuration to the experiment.

## Supported configurations and tests

Python 3.12–3.13 is supported; [constraints-tested.txt](constraints-tested.txt) records the Python 3.12 CPU environment. The primary training route is vision with torchvision ViT. Optional DINO models, ADC/C3-SL/Rand-Top-K baselines and audio modules are experimental extensions; audio requires separate external dependencies. FLOP profiling counts operators, while latency and energy require hardware measurements.

```bash
python -m pytest -q
ae-psl-smoke --seed 42 --output results/smoke.json
```

Tests cover serialized-gradient equivalence, frozen compressors/backbones, local adaptation, global and personalized checkpoint selection, dataset/cache integrity and campaign resume. Small ViTs and prepared image/LEAF fixtures exercise the complete GA → CSA → DFT route for CIFAR-100, Food101, SUN397, ImageNet100 and FEMNIST without full corpus or pretrained-weight downloads.
