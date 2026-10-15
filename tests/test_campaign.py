import json
from pathlib import Path

import pytest
import torch

from ae_psl.orchestrator import setup_orchestrator_parser
from ae_psl.utils.file_utils import save_object_to_json
from ae_psl.utils.orchestrator_argument_utils import (
    namespace_to_global_args_and_search_space,
)


def test_json_failure_does_not_destroy_previous_result(tmp_path):
    path = tmp_path / "result.json"
    save_object_to_json({"previous": True}, path)
    with pytest.raises(TypeError):
        save_object_to_json({"bad": object()}, path)
    assert json.loads(path.read_text()) == {"previous": True}


@pytest.mark.parametrize("failure", ["raises", "returns_without_saving", "write_error", "keyboard_interrupt"])
def test_interrupted_epoch_result_is_not_a_completed_campaign_job(
    tmp_path, monkeypatch, failure
):
    from ae_psl import orchestrator

    args = setup_orchestrator_parser().parse_args(
        [
            "--dataset",
            "cifar100",
            "--nr_of_epochs",
            "2",
            "--experiments_dir",
            str(tmp_path),
        ]
    )
    global_args, search = namespace_to_global_args_and_search_space(args)
    folder, _manifest = orchestrator.generate_manifest(
        "review", str(tmp_path), global_args, search
    )
    args.resume_from_manifest = str(Path(folder) / "manifest.json")
    output = Path(folder) / "logs" / "run_default.json"
    save_object_to_json(
        {"finetune_results": {"epochs": [0], "train_metric": [0.5]}}, output
    )

    def interrupted_training(*args):
        if failure == "write_error":
            save_object_to_json({"bad": object()}, output)
        if failure == "raises":
            raise RuntimeError("incomplete job was retried")
        if failure == "keyboard_interrupt":
            raise KeyboardInterrupt

    monkeypatch.setattr(orchestrator, "run_2_stage_mpsl", interrupted_training)
    expected = KeyboardInterrupt if failure == "keyboard_interrupt" else RuntimeError
    with pytest.raises(expected):
        orchestrator.run_orchestrator(args)
    status = json.loads(output.with_suffix(".status.json").read_text())
    assert status["state"] == ("interrupted" if failure == "keyboard_interrupt" else "failed")
    assert "completion" not in json.loads(output.read_text())


@pytest.mark.parametrize("personalized", [False, True])
def test_best_latest_final_checkpoints_and_test_loss(
    tmp_path, monkeypatch, personalized
):
    if personalized:
        from ae_psl.main_AE_PSL_local_evaluation import finetune_mpsl
    else:
        from ae_psl.main_AE_PSL_global_evaluation import finetune_mpsl
    monkeypatch.setenv("MODEL_WEIGHTS_DIR", str(tmp_path))
    args, _ = namespace_to_global_args_and_search_space(
        setup_orchestrator_parser().parse_args(["--dataset", "food101"])
    )
    args.update(
        nr_of_epochs=2,
        nr_of_clients=2,
        client_specific_alignment=False,
        encoder_only_client_specific_alignment=False,
        save_model_after_each_epoch=True,
        save_final_model=True,
        save_file_name=str(tmp_path / "model"),
    )
    clients = {i: torch.nn.ParameterDict({'weight': torch.nn.Parameter(torch.zeros(1))}) for i in (0, 1)}
    server = torch.nn.ParameterDict({'weight': torch.nn.Parameter(torch.zeros(1))})
    optimizer = torch.optim.SGD(server.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1)
    validation = (
        {0: ["validation"], 1: ["validation"]} if personalized else ["validation"]
    )
    test = {0: ["test"], 1: ["test"]} if personalized else ["test"]

    class Trainer:
        # Controlled epoch weights isolate checkpoint selection from learning dynamics.
        def train_epoch(self, **kwargs):
            epoch = kwargs["epoch_nr"]
            with torch.no_grad():
                server.weight.fill_(epoch + 1)
                for i, model in clients.items():
                    model.weight.fill_(10 * i + epoch + 1)
            optimizer.step()
            return 1.0, 0.5, {0: 1, 1: 1}

        def test_epoch_global_evaluation(self, client, model, loader, *unused):
            return model.weight.item(), 0.5

        def test_epoch_distributed(self, models, model, loaders, *unused):
            return model.weight.item(), 0.5, {i: {"samples": 1} for i in models}

    result = finetune_mpsl(
        args,
        "cpu",
        server,
        clients,
        Trainer(),
        [],
        validation,
        test,
        {},
        {},
        {},
        {},
        optimizer,
        scheduler,
        True,
        1,
        {},
        "early_saving",
    )
    assert server.weight.item() == 1  # Epoch zero was the best, epoch one was worse.
    assert result.final_test_loss == 1
    assert result.last_completed_epoch == 1
    assert result.best_validation_epoch == 0
    final = torch.load(tmp_path / "model.pth", weights_only=True)
    latest = torch.load(tmp_path / "model.latest.pth", weights_only=True)
    assert final["server_model"]["weight"].item() == 1
    assert latest["server_model"]["weight"].item() == 2
    if personalized:
        assert set(final["client_models"]) == {0, 1}
        assert final["client_models"][0]["weight"].item() == 1
        assert final["client_models"][1]["weight"].item() == 11


def test_ga_uses_supplied_training_partition(tmp_path, monkeypatch):
    from torch.utils.data import TensorDataset

    from ae_psl.ae_trainers import ae_trainer
    from ae_psl.smoke import create_system

    args, _ = namespace_to_global_args_and_search_space(
        setup_orchestrator_parser().parse_args(["--dataset", "food101"])
    )
    paper_args, base, _ = create_system()
    args.update(paper_args)
    args.update(
        ae_pretrain_dataset="synthetic",
        ae_force_retrain=True,
        ae_pretrain_epochs=1,
        ae_pretrain_batch_size=2,
        num_workers_activation_loading=0,
        num_workers_ae_pretraining=0,
    )
    monkeypatch.setenv("AE_WEIGHTS_DIR", str(tmp_path / "ae"))
    monkeypatch.chdir(tmp_path)

    class Data:
        train_ds = TensorDataset(torch.randn(2, 3, 224, 224), torch.zeros(2))

        def load_partition(self, partition_id):
            raise AssertionError("GA must use all client training examples")

        def load_test_set(self):
            return TensorDataset(torch.randn(1, 3, 224, 224), torch.zeros(1))

        def get_collate_fn(self):
            return None

    def unexpected_reload(**kwargs):
        raise AssertionError(
            "GA must not reload and include the held-out validation split"
        )

    monkeypatch.setattr(ae_trainer.available_datasets, "load_data", unexpected_reload)
    _model, metrics = ae_trainer.get_auto_encoder(args, base, "cpu", Data())
    assert metrics["epochs"] == [0]
    assert metrics["final_test_loss"] >= 0


def test_completion_evidence_rejects_partial_and_accepts_proven_legacy(tmp_path):
    from ae_psl.utils.campaign_state import completed_result

    path = tmp_path / "job.json"
    args = {"nr_of_epochs": 2}
    metrics = {
        "epochs": [0],
        "train_metric": [0.5],
        "final_test_loss": 1.0,
        "final_test_accuracy": 0.5,
    }
    save_object_to_json({"finetune_results": metrics}, path)
    assert completed_result(path, args) is None
    metrics.update(epochs=[0, 1], train_metric=[0.5, 0.5])
    save_object_to_json({"finetune_results": metrics, "ae_pretrain_results": {}}, path)
    assert completed_result(path, args)["legacy_completion_evidence"]
    metrics.update(last_completed_epoch=0)
    completion = {
        "state": "completed",
        "stage": "dft",
        "requested_epochs": 2,
        "last_completed_epoch": 0,
        "stopped_early": False,
    }
    save_object_to_json({"finetune_results": metrics, "completion": completion}, path)
    assert completed_result(path, args) is None
    completion["stopped_early"] = True
    save_object_to_json({"finetune_results": metrics, "completion": completion}, path)
    assert completed_result(path, args)["last_completed_epoch"] == 0


@pytest.mark.parametrize("ga_only", [False, True])
def test_primary_orchestrator_runs_real_phases_and_resumes(
    tmp_path, monkeypatch, ga_only
):
    from types import SimpleNamespace

    from torch.utils.data import Subset, TensorDataset

    from ae_psl import main_AE_PSL_global_evaluation as experiment
    from ae_psl import orchestrator
    from ae_psl.models.vision_transformer.base.vision_transformer_base import (
        VisionTransformerBase,
    )

    torch.set_num_threads(1)
    monkeypatch.chdir(tmp_path)
    args = setup_orchestrator_parser().parse_args(
        [
            "--dataset",
            "food101",
            "--ae_pretrain_dataset",
            "food101",
            "--nr_of_epochs",
            "1",
            "--ae_pretrain_epochs",
            "1",
            "--batch_size",
            "2",
            "--ae_pretrain_batch_size",
            "2",
            "--nr_of_clients",
            "2",
            "--split_layer",
            "5",
            "--ae_latent_dim",
            "64",
            "--ae_force_retrain",
            "true",
            "--num_workers_finetuning",
            "0",
            "--num_workers_ae_pretraining",
            "0",
            "--num_workers_activation_loading",
            "0",
            "--save_final_model",
            "true",
            "--ae_pretrain_only",
            str(ga_only),
            "--ae_weights_dir",
            str(tmp_path / "ae"),
            "--experiments_dir",
            str(tmp_path / "campaign"),
        ]
    )

    class FixtureData:
        def __init__(self):
            self.train_ds = TensorDataset(
                torch.randn(4, 3, 224, 224), torch.arange(4) % 3
            )
            self.partitioner = SimpleNamespace(
                _partition_id_to_indices={0: [0], 1: [1, 2, 3]},
                _determine_partition_id_to_indices_if_needed=lambda: None,
            )

        def load_partition(self, partition_id):
            return Subset(
                self.train_ds, self.partitioner._partition_id_to_indices[partition_id]
            )

        def load_validation_set(self):
            return None

        def load_test_set(self):
            return TensorDataset(torch.randn(1, 3, 224, 224), torch.arange(1))

        def get_collate_fn(self):
            return None

    def paper_backbone(args, device):
        return VisionTransformerBase(
            "vit_b_32", True, 16, 32, 101, device, pretrained=False
        )

    # Replace only external data/checkpoint acquisition. All phase orchestration,
    # trainers, optimizers, model splitting, cache/output writes and resume are real.
    monkeypatch.setattr(
        experiment.datasets, "load_data", lambda **kwargs: FixtureData()
    )
    monkeypatch.setattr(experiment, "get_base_model", paper_backbone)
    orchestrator.run_orchestrator(args)
    manifest = next((tmp_path / "campaign").glob("*/manifest.json"))
    output = manifest.parent / "logs/run_default.json"
    result = json.loads(output.read_text())
    assert result["completion"]["state"] == "completed"
    assert result["completion"]["last_completed_epoch"] == 0
    assert result["ae_pretrain_results"]["epochs"] == [0]
    if not ga_only:
        assert result["finetune_results"]["final_test_loss"] >= 0
        assert "-1" in result["finetune_results"]["client_batch_mse"]["0"]  # CSA
        assert (manifest.parent / "logs/run_default.pth").exists()
    saved = output.read_bytes()
    args.resume_from_manifest = str(manifest)
    monkeypatch.setattr(
        experiment.datasets,
        "load_data",
        lambda **kwargs: pytest.fail("Completed job restarted"),
    )
    orchestrator.run_orchestrator(args)
    assert output.read_bytes() == saved
