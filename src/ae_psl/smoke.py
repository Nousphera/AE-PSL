"""Offline GA → CSA → DFT using the paper ViT-B/32, initialized from scratch."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ae_psl.ae_trainers.implementations.classification.centralized_ae_trainer import (
    CentralizedAETrainer,
)
from ae_psl.models.auto_encoder import initialize_AE
from ae_psl.models.vision_transformer.base.vision_transformer_base import (
    VisionTransformerBase,
)
from ae_psl.models.vision_transformer.implementations.unimodal.image_classification.models import (
    get_split_model,
)
from ae_psl.trainers.implementations.classification.sasl_trainer import SASLTrainer
from ae_psl.trainers.implementations.experiment_results import ExperimentResults
from ae_psl.utils.fl_utils import fed_avg
from ae_psl.utils.scheduler_utils import (
    get_optimzer_and_scheduler_for_seperate_ae_finetuning,
)


def configuration(seed=42):
    return dict(
        seed=seed,
        model="vit_b_32",
        initialization="from_scratch",
        image_size=224,
        patch_size=32,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        split_layer=5,
        num_classes=100,
        lora_rank=16,
        lora_alpha=32,
        nr_of_clients=2,
        ae_type="2_layer_MLP_(d_d_d_z)",
        ae_latent_dim=64,
        compression_method="ae",
        ae_finetune_lr=0.001,
        start_lr=0.0001,
        scheduler="constant",
        dataset="synthetic",
        concurrent_mse_alignment=False,
        ae_freeze_encoder_during_finetuning=True,
        ae_freeze_decoder_during_finetuning=True,
        freeze_clientside=False,
        no_compression_batches=0,
        small_test_run=False,
        compress_cls_token=False,
    )


def create_system(seed=42):
    """Build fresh actual split modules with no asset downloads."""
    torch.manual_seed(seed)
    args = configuration(seed)
    base = VisionTransformerBase("vit_b_32", True, 16, 32, 100, "cpu", pretrained=False)
    ae = initialize_AE(args, 768)
    return args, base, ae


def state_hashes(module):
    return {
        k: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()
        for k, v in module.state_dict().items()
    }


def run(seed=42):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args, base, ae = create_system(seed)
    base.eval()
    base_before = state_hashes(base)
    public_x = torch.randn(2, 3, 224, 224)
    public_features = base.retrieve_split_layer_activations(public_x, 5)
    loader = DataLoader(TensorDataset(public_features), batch_size=1, shuffle=False)
    ga_before = state_hashes(ae)
    ga_optimizer = torch.optim.Adam(ae.parameters(), lr=0.0001)
    ga_loss = CentralizedAETrainer().train_epoch(
        experiment_results=None,
        base_model=base,
        auto_encoder=ae,
        split_layer=5,
        device="cpu",
        dataloader=loader,
        epoch_nr=0,
        optimizer=ga_optimizer,
        loss_fn=torch.nn.MSELoss(),
    )
    ga_changed = state_hashes(ae) != ga_before
    base_unchanged = state_hashes(base) == base_before

    client, server, _ = get_split_model(base, ae, 5, 100, "cpu", False)
    clients = {i: copy.deepcopy(client) for i in range(2)}
    # Distinct synthetic distributions and uneven last batches exercise aggregation.
    data = {
        i: (torch.randn(1 + i, 3, 224, 224) + i * 0.5, torch.arange(1 + i) % 100)
        for i in clients
    }
    batches = [
        {
            i: (x[start : start + 1], y[start : start + 1])
            for i, (x, y) in data.items()
            if start < len(x)
        }
        for start in (0, 1)
    ]
    optimizers = {
        i: get_optimzer_and_scheduler_for_seperate_ae_finetuning(c, args)
        for i, c in clients.items()
    }
    main = {i: value[0] for i, value in optimizers.items()}
    ae_opts = {i: value[2] for i, value in optimizers.items()}
    main_schedulers = {i: value[1] for i, value in optimizers.items()}
    ae_schedulers = {i: value[3] for i, value in optimizers.items()}
    server_opt = torch.optim.Adam(server.parameters(), lr=args["start_lr"])
    results = ExperimentResults("none")
    trainer = SASLTrainer()
    csa_loss = trainer.client_specific_alignment(
        "cpu",
        server,
        server_opt,
        batches,
        clients,
        True,
        main,
        main_schedulers,
        ae_opts,
        ae_schedulers,
        len(batches),
        results,
        0,
        args,
    )
    local_encoders_distinct = any(
        not torch.equal(v, clients[1].ae_module.patch_encoder.state_dict()[k])
        for k, v in clients[0].ae_module.patch_encoder.state_dict().items()
    )
    for key, value in server.ae_module.patch_decoder.state_dict().items():
        torch.testing.assert_close(
            value,
            sum(c.ae_module.patch_decoder.state_dict()[key] for c in clients.values())
            / len(clients),
        )
    frozen = {i: state_hashes(c.ae_module) for i, c in clients.items()}
    frozen_server = state_hashes(server.ae_module)
    before_clients = {i: state_hashes(c) for i, c in clients.items()}
    before_server = state_hashes(server)
    train_loss, train_acc, counts = trainer.train_epoch(
        "cpu",
        server,
        server_opt,
        batches,
        clients,
        True,
        main,
        main_schedulers,
        ae_opts,
        ae_schedulers,
        len(batches),
        results,
        1,
        args,
    )
    ae_frozen = all(state_hashes(c.ae_module) == frozen[i] for i, c in clients.items())
    ae_frozen &= state_hashes(server.ae_module) == frozen_server
    lora_updated = all(
        any(
            "lora_" in k and v != before_clients[i][k]
            for k, v in state_hashes(c).items()
        )
        for i, c in clients.items()
    )
    base_frozen_dft = all(
        v == before_clients[i][k]
        for i, c in clients.items()
        for k, v in state_hashes(c).items()
        if "lora_" not in k and "ae_module" not in k
    )
    base_frozen_dft &= all(
        v == before_server[k]
        for k, v in state_hashes(server).items()
        if "lora_" not in k and not k.startswith(("heads.", "ae_module."))
    )
    lora_gradients = all(
        p.grad is not None
        for model in [*clients.values(), server]
        for name, p in model.named_parameters()
        if "lora_" in name
    )
    # Test-set labels never select a checkpoint or influence training.
    test_x, test_y = torch.randn(2, 3, 224, 224), torch.arange(2) % 100
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=1)
    averaged = fed_avg(
        clients, {i: count / sum(counts.values()) for i, count in counts.items()}
    )
    global_loss, global_acc = trainer.test_epoch_global_evaluation(
        averaged, server, test_loader, "cpu", None, 1, False, False
    )
    local_loaders = {
        i: DataLoader(TensorDataset(test_x[i::2], test_y[i::2]), batch_size=1)
        for i in clients
    }
    local_loss, local_acc, local_metrics = trainer.test_epoch_distributed(
        clients, server, local_loaders, "cpu", None, 1, False, True
    )
    checks = dict(
        ga_updates_ae=ga_changed,
        ga_freezes_backbone=base_unchanged,
        csa_retains_distinct_local_encoders=local_encoders_distinct,
        dft_freezes_ae=ae_frozen,
        dft_freezes_base=base_frozen_dft,
        dft_updates_each_client_lora=lora_updated,
        all_lora_receive_gradients=lora_gradients,
    )
    if not all(checks.values()):
        raise RuntimeError(f"Protocol invariant failed: {checks}")
    checkpoint = {
        f"client_{i}.{k}": v.cpu().numpy()
        for i, c in clients.items()
        for k, v in c.state_dict().items()
    }
    checkpoint.update(
        {f"server.{k}": v.cpu().numpy() for k, v in server.state_dict().items()}
    )
    hasher = hashlib.sha256()
    for key in sorted(checkpoint):
        hasher.update(checkpoint[key].tobytes())
    digest = hasher.hexdigest()
    metrics = dict(
        config=args,
        checks=checks,
        ga_mse=ga_loss,
        csa_mse=csa_loss,
        train_loss=train_loss,
        train_accuracy=train_acc,
        global_loss=global_loss,
        global_accuracy=global_acc,
        local_loss=local_loss,
        local_accuracy=local_acc,
        local_clients=local_metrics,
        sample_counts=counts,
        state_sha256=digest,
        communication=results.to_json(),
        environment={
            "python": platform.python_version(),
            **{
                p: importlib.metadata.version(p)
                for p in ("torch", "torchvision", "numpy", "tqdm")
            },
        },
    )
    return metrics, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/smoke.json"))
    args = parser.parse_args()
    checkpoint_path = args.output.with_suffix(".npz")
    if args.output.exists() or checkpoint_path.exists():
        parser.error("Output exists; choose a new --output path")
    metrics, checkpoint = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("xb") as f:
        np.savez_compressed(f, **checkpoint)
    with args.output.open("x") as f:
        json.dump(metrics, f, indent=2)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "checkpoint": str(checkpoint_path),
                "state_sha256": metrics["state_sha256"],
                "checks": metrics["checks"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
