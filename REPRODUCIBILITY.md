# Reproducibility and implementation mapping

## Paper-to-code map

The reference is arXiv:2607.17913, *AutoEncoder-Compressed Parallel Split Learning for Pre-Trained Model Fine-Tuning*. Paths below are relative to `src/ae_psl/`.

| Paper component | Implementation | Verification |
|---|---|---|
| §4.1.1 public-feature GA | `ae_trainers/ae_trainer.py`, `centralized_ae_trainer.py` | Actual ViT feature extraction; AE changes while backbone stays fixed |
| §4.1.2 local CSA | `trainers/implementations/classification/sasl_trainer.py` | Unconditional local optimizer steps; distinct retained encoders; uniform decoder-only average, including unequal client dataset sizes |
| §4.2 frozen encoder/decoder DFT | Same trainer, split image-classification models | Encoder stays differentiable in its input; AE parameters unchanged; all installed LoRA parameters receive gradients |
| CLS preservation | `models/auto_encoder.py`, split models | Exact CLS identity and CLS gradient propagation |
| Single server backward | `transport.py`, main trainer | Serialized detached tensors; client logit gradients; one server backward; full updates equal direct autograd |
| §5.2 global evaluation | `test_epoch_global_evaluation` | Average client model; bypass both AE halves; example-weighted metrics |
| §5.2 local evaluation | `test_epoch_distributed` | Client's own encoder; shared decoder; macro average across clients, with per-client counts and metrics |
| LoRA | `models/lora_layer.py`, `vision_transformer_base.py` | Biasless support; real gradient/update checks |

The default torchvision ViT adapts its MLP Linear layers plus classifier. The original attempted `MultiheadAttention.out_proj` adapters were never executed because PyTorch reads its weight directly; they are now omitted. No Q/K/V adaptation is implied. The paper gives rank 16 and alpha 32 without an exact target-module list; this release preserves the effective original torchvision target selection.

The source AE uses LayerNorm/GELU and 0.01 encoder dropout during alignment. DFT sets frozen AEs to evaluation mode, disabling stochastic dropout while retaining input gradients. Reconstruction MSE retains the source's full-token reduction; the unchanged CLS contributes zero error, scaling patch-only mean MSE by `patches / (patches + 1)`. This affects the loss scale, not which tokens are reconstructed.

## Transport and claims

This is a serialized, single-process research implementation, not a network orchestration service. Each client's latent tokens and CLS token are encoded as float32 bytes with JSON shape/dtype metadata. The server receives independent leaf tensors. Logits return through the same boundary; a client computes cross entropy using its labels and returns `n_client / n_global_batch * dLoss/dLogits`. The server concatenates gradients in activation order and performs one backward pass. Latent/CLS gradients return to the original client graphs. Tests compare every updated parameter against direct joint autograd on uneven client batches.

Paper §4.2 writes a sum of client losses. This release retains the source's sample-count-weighted average of client mean cross entropies, including unequal final batches; its equivalence tests use that precise objective. It does not claim numerical identity to an unweighted sum of client means.

The paper describes scalar client loss transmission. Scalars alone cannot reconstruct a server-side derivative after a real process boundary. This release explicitly transmits logit gradients, adds their cost, and sends scalar loss telemetry. Therefore its communication totals are not directly the paper's reported ratios. DFT records exact message byte lengths, including shape/dtype headers, for these logical messages. It excludes transport protocols, encryption, model/optimizer distribution, GA dataset movement and wall-clock networking costs. CSA records raw decoder tensor bytes, separately from DFT framing; it does not claim a network broadcast. Units in legacy result keys ending `_in_mb` are MiB (bytes / 2^20).

The `server_batch_mse` field is a simulator-only diagnostic that compares the original client activations with server reconstructions locally. It does not affect optimization and its shared-memory inspection is excluded from protocol message counts; a distributed backend must omit it or account for the additional exchange. Scalar metrics used for server loss reporting are taken from the serialized telemetry.

Labels remain in the local client-loss calculation. This does not establish label privacy: gradients can reveal information. No encryption, secure aggregation, privacy proof, multi-host deployment, or production transport security is supplied. A Flower wrapper would not solve this split-backpropagation protocol and is not included.

## Datasets and experiments

The `ae-psl` command is the original experiment orchestrator with modern package imports. Default ViT-B/32 uses torchvision `ViT_B_32_Weights.IMAGENET1K_V1` (the current default of the pinned torchvision release). Full experiments need substantial compute and dataset storage. The CPU smoke does not download or validate those assets.

- **CIFAR-100 and Food101:** torchvision train/test splits; auto-download on requested experiment execution. Client partitions use Dirichlet alpha 0.5 with the requested seed for `--dataset_split_type noniid`.
- **SUN397:** inherited deterministic 80/20 random split, fixed seed 42, over torchvision SUN397. This is not the official SUN397 partition protocol. The paper does not fully identify its split manifest; results require confirming this convention.
- **ImageNet100 for GA:** prepare `data/datasets/imagenet100/train/<class>/*` and `data/datasets/imagenet100/test/<class>/*`, using the same 100 class folders. Data are not redistributed or downloaded. Optional `Labels.json` maps names only. The original exact 100-class list and image manifest are unavailable; supply and record your own and do not call results an exact reproduction. Override root with `--torch_data_dir`.
- **FEMNIST local evaluation:** install `python -m pip install '.[femnist]'`, then use `--dataset femnist`. The inherited loader uses `flwrlabs/femnist`, writer partitions, and 10% writer subsampling. It has not been exercised in this release; verify dataset revision, class mapping and writer/train/test manifests before reporting results.

Default paper-shaped recipe: ViT-B/32, split 5, two-layer MLP `(d,d,d_z)`, latent dimensions 140/64/24, GA 30 epochs on ImageNet100, CSA one epoch, DFT 10 epochs, LoRA rank 16/alpha 32, Adam 1e-4, global batch 125, 5 or 25 clients, seeds 42/43/44. A latent dimension equal to hidden size selects an identity compressor. GA validation and DFT validation default to disabled; test accuracy does not select a checkpoint. `--save_final_model true` writes final model state. See `run_basic_example.sh` for an explicit sweep and `ae-psl --help` for flags.

Campaigns produce a dated manifest, per-job result JSON, `.status.json` and optional model checkpoints. A successful result has an explicit `completion` record containing its stage, requested epochs, zero-based last completed epoch and early-stop state. In-progress epoch JSON is never sufficient for skipping a job. Resume with `--resume_from_manifest PATH`; completed outputs are verified before skipping. Legacy combined outputs are accepted only with final metrics, the complete requested epoch history and the final AE-results section; ambiguous older outputs rerun. GA-only jobs have their own completion stage.

JSON records are written to a temporary file, flushed and atomically replaced; serialization and filesystem errors propagate. A training return without a saved final result is a failed job. Campaign resume restarts incomplete jobs from their beginning, not from optimizer state. It may reuse a compatible GA cache. Failures cause a nonzero exit after the campaign; interruption exits unsuccessfully. Choose separate directories for independent campaigns.

With validation, `.best.pth` is an isolated temporary best-validation snapshot. Optional `.latest.pth` records the latest epoch and cannot overwrite it. Final evaluation reloads the best snapshot and records both test loss and accuracy; `--save_final_model true` writes that evaluated state to `.pth`. A personalized checkpoint contains `client_models` keyed by every client ID plus the shared `server_model`, so all local encoders and adapters restore together. Global checkpoints retain `client_model` plus `server_model`. No checkpoint includes optimizer-resume state. The result records `best_validation_epoch` separately from `last_completed_epoch`.

GA cache schema 3 binds seed, training settings, model state hash, model/split, absolute dataset root and whether the existing downstream training split is supplied. In the downstream-GA ablation, all supplied training examples are used and held-out validation examples remain excluded. They do not hash every dataset file: when changing images or class lists at the same path, remove the corresponding local activation/AE cache or use a fresh output/data directory. Legacy caches without the expanded signature are not silently reused. `--ae_specific_weights_path` explicitly overrides discovery; confirm that checkpoint's provenance yourself.

Optional DINO models, historical ADC/C3-SL/Rand-Top-K baselines, and audio extensions remain in source. They are not validated benchmark reproductions. Audio imports no longer block vision setup; the old audio path depends on historical external packages and is unsupported here. `--profile_flops` retains research profiling; operator totals do not establish measured latency, energy, or the paper's hardware GFLOP results.

## Validated release scope

Supported Python is 3.12–3.13; the tested NumPy 2.5.4 constraint requires Python >=3.12. Local validation used Python 3.12 / macOS arm64 CPU, torch 2.8.0, torchvision 0.23.0, NumPy 2.5.4. Exact pins are in `constraints-tested.txt`. The test suite covers regressions, byte-boundary independence, gradient equivalence, fixed compressors/backbones, nontrivial local adaptation, evaluation routes, cache identity, and deterministic repeated seeds. The no-download integration run uses two real transformer blocks, hidden dimension 16, four patch tokens, latent dimension 4, two clients with 5/6 training examples, 12 public examples and seven test examples.

A separate integration check executes the primary campaign orchestrator through actual GA → CSA → DFT and output/resume handling, replacing only external dataset and pretrained-backbone acquisition with tiny fixtures. GA-only completion/resume is also exercised. Checkpoint selection tests cover global and personalized paths. These checks do not exercise the FEMNIST provider.

Neither the paper's accuracy tables nor GPU-scale sweeps were rerun. Changes to CSA, metric averaging, dropout mode and transport fix demonstrable discrepancies; they also mean outputs can differ from historical runs. Existing dataset loaders and ablation baselines require further qualification before comparing reported numbers.
