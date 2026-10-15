"""Preparation contract and actual primary trainers, with small decoded images."""
import io
import json
import os
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torchvision import transforms

from ae_psl.prepare_data import prepare_images, prepare_femnist, validate_manifest, extract_tar_checked
from ae_psl.available_datasets.prepared import femnist_partitions


def image_source(root, per_class=16):
    for split, count in (("train", per_class), ("test", 2)):
        for label in range(3):
            folder = root / split / str(label)
            folder.mkdir(parents=True)
            for index in range(count):
                pixels = np.random.default_rng(label * 1000 + index).integers(0, 256, (16, 16, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(folder / f"{index}.png")
    return root


def leaf_source(root, writers=5):
    for split, count in (("train", 4), ("test", 2)):
        values = {}
        for writer in range(writers):
            values[str(writer)] = {
                "x": [np.random.default_rng(writer * 1000 + i).random(784).tolist() for i in range(count)],
                "y": [i % 3 for i in range(count)],
            }
        folder = root / split
        folder.mkdir(parents=True)
        (folder / "all.json").write_text(json.dumps({"user_data": values}))
    return root


@pytest.mark.parametrize("dataset", ["cifar100", "food101", "sun397", "imagenet100"])
def test_prepared_images_reach_actual_dataset_loader(tmp_path, monkeypatch, dataset):
    from ae_psl import available_datasets
    root = tmp_path / "prepared"
    manifest = prepare_images(dataset, root, image_source(tmp_path / "source"), allow_subset=True)
    value = validate_manifest(manifest, True)
    assert len(value["records"]) == 54
    monkeypatch.setenv("TORCH_DATA_DIR", str(root))
    monkeypatch.setattr(available_datasets, "get_default_vit_transforms", lambda train: transforms.ToTensor())
    data = available_datasets.load_data(name="imagenet" if dataset == "imagenet100" else dataset,
                                        num_partitions=2, global_args={})
    assert len(data.train_ds) == 48
    assert sum(len(data.load_partition(i)) for i in range(2)) == 48
    assert data.load_partition(0)[0][0].shape == (3, 16, 16)
    assert len(data.load_test_set()) == 6
    with pytest.raises(FileExistsError):
        prepare_images(dataset, root, tmp_path / "source", allow_subset=True)
    image = manifest.parent / value["records"][0]["path"]
    image.write_bytes(b"not an image")
    with pytest.raises(Exception):
        validate_manifest(manifest, True)


def test_import_refuses_overlap_and_preserves_no_partial_output(tmp_path):
    source = image_source(tmp_path / "source", 1)
    (source / "test/0/0.png").unlink()
    os.link(source / "train/0/0.png", source / "test/0/0.png")
    with pytest.raises(ValueError, match="Repeated source file"):
        prepare_images("food101", tmp_path / "prepared", source, allow_subset=True)
    assert not (tmp_path / "prepared/food101").exists()


@pytest.mark.parametrize("name,kind", [("../escape", "file"), ("/escape", "file"), ("link", "symlink")])
def test_archive_rejects_unsafe_members(tmp_path, name, kind):
    archive = tmp_path / "archive.tar"
    with tarfile.open(archive, "w") as stream:
        member = tarfile.TarInfo(name)
        if kind == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = "../escape"
            stream.addfile(member)
        else:
            member.size = 1
            stream.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Unsafe archive"):
        extract_tar_checked(archive, tmp_path / "output")
    assert not (tmp_path / "escape").exists()


def test_femnist_keeps_remainder_and_keys_cache_by_source_and_validation(tmp_path):
    manifest = prepare_femnist(tmp_path / "prepared", leaf_source(tmp_path / "leaf"))
    a = femnist_partitions(manifest, 2, 42, 1, .25)
    assert a == femnist_partitions(manifest, 2, 42, 1, .25)
    assert sorted(w for p in a["partitions"] for w in p["writers"]) == [str(i) for i in range(5)]
    rows = validate_manifest(manifest)["records"]
    all_ids = []
    for part in a["partitions"]:
        selected = part["train"] + part["val"] + part["test"]
        assert len(selected) == len(set(selected))
        assert all(rows[i]["split"] == "train" for i in part["train"] + part["val"])
        assert all(rows[i]["split"] == "test" for i in part["test"])
        all_ids.extend(selected)
    assert sorted(all_ids) == list(range(30))
    assert a["signature"] != femnist_partitions(manifest, 2, 42, 1, .5)["signature"]
    value = json.loads(manifest.read_text())
    value["provenance"]["revision_note"] = "changed source identity"
    manifest.write_text(json.dumps(value))
    assert a["signature"] != femnist_partitions(manifest, 2, 42, 1, .25)["signature"]
    assert len(list((manifest.parent / "partitions").glob("*.json"))) == 3


@pytest.mark.parametrize("dataset", ["cifar100", "food101", "sun397", "imagenet", "femnist"])
def test_prepared_images_run_primary_ga_csa_dft(tmp_path, monkeypatch, dataset):
    from torchvision.models.vision_transformer import VisionTransformer
    from ae_psl import available_datasets, orchestrator
    from ae_psl import main_AE_PSL_global_evaluation as global_experiment
    from ae_psl import main_AE_PSL_local_evaluation as local_experiment
    from ae_psl.models.vision_transformer.base.vision_transformer_base import VisionTransformerBase

    torch.set_num_threads(1)
    root = tmp_path / "prepared"
    if dataset == "femnist":
        prepare_femnist(root, leaf_source(tmp_path / "leaf"))
    else:
        prepare_images("imagenet100" if dataset == "imagenet" else dataset, root, image_source(tmp_path / "source"), allow_subset=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(available_datasets, "get_default_vit_transforms", lambda train: transforms.ToTensor())
    monkeypatch.setattr(available_datasets, "get_vit_mnist_transforms", lambda: transforms.Compose([
        transforms.Resize((16, 16)), transforms.ToTensor()]))

    def small_backbone(args, device):
        count = {"femnist": 62, "food101": 101, "sun397": 397, "imagenet": 100, "cifar100": 100}[dataset]
        size = 224 if dataset == "cifar100" else 16
        backbone = VisionTransformer(image_size=size, patch_size=size // 2, num_layers=2,
            num_heads=2, hidden_dim=16, mlp_dim=32, num_classes=count)
        return VisionTransformerBase("vit_b_32", True, 2, 4, count, device, backbone=backbone)

    monkeypatch.setattr(global_experiment, "get_base_model", small_backbone)
    monkeypatch.setattr(local_experiment, "get_base_model", small_backbone)
    args = orchestrator.setup_orchestrator_parser().parse_args([
        "--dataset", dataset, "--ae_pretrain_dataset", dataset,
        "--torch_data_dir", str(root), "--dataset_split_type", "iid",
        "--nr_of_epochs", "1", "--ae_pretrain_epochs", "1", "--batch_size", "4",
        "--ae_pretrain_batch_size", "2", "--nr_of_clients", "2", "--split_layer", "1",
        "--ae_latent_dim", "4", "--ae_force_retrain", "true", "--femnist_fraction", "1",
        "--num_workers_finetuning", "0", "--num_workers_ae_pretraining", "0",
        "--num_workers_activation_loading", "0", "--test_num_workers", "0",
        "--val_mode", "early_saving", "--val_split", "0.25",
        "--save_final_model", "true", "--ae_weights_dir", str(tmp_path / "ae"),
        "--experiments_dir", str(tmp_path / "campaign"),
    ])
    orchestrator.run_orchestrator(args)
    output = next((tmp_path / "campaign").glob("*/logs/run_default.json"))
    result = json.loads(output.read_text())
    assert result["completion"]["state"] == "completed"
    assert result["ae_pretrain_results"]["epochs"] == [0]
    assert result["finetune_results"]["final_test_loss"] >= 0
    assert "-1" in result["finetune_results"]["client_batch_mse"]["0"]
    assert output.with_suffix(".pth").exists()
    if dataset == "femnist":
        assert len(result["finetune_results"]["distributed_eval_final"]) == 2
        cache = next((root / "femnist/partitions").glob("*.json"))
        assert sum(len(p["writers"]) for p in json.loads(cache.read_text())["partitions"]) == 5
    signatures = list((tmp_path / "ae").rglob("signature.json"))
    assert len(signatures) == 1
    signature = json.loads(signatures[0].read_text())
    assert len(signature["prepared_manifest_sha256"]) == 64
    if dataset == "femnist":
        assert len(signature["supplied_split_sha256"]) == 64


@pytest.mark.parametrize("dataset", ["cifar100", "food101", "sun397"])
def test_official_download_recipe_uses_checked_acquisition(tmp_path, monkeypatch, dataset):
    from ae_psl import prepare_data
    acquired = []
    monkeypatch.setattr(prepare_data, "acquire_official", lambda name, root, archive=None: acquired.append(name) or {"url": "fixture://official"})

    class Provider:
        classes = ["a", "b", "c"]
        def __init__(self, root, download=False, **kwargs):
            assert download is False  # Unchecked torchvision extractall must never run.
            self._image_files = [Path(root) / str(i) for i in range(10)]
            self._labels = [i % 3 for i in range(10)]
        def __len__(self):
            return 10
        def __getitem__(self, index):
            return Image.new("RGB", (4, 4), (index, 0, 0)), self._labels[index]

    monkeypatch.setattr(prepare_data.datasets, {"cifar100": "CIFAR100", "food101": "Food101", "sun397": "SUN397"}[dataset], Provider)
    manifest = prepare_images(dataset, tmp_path / "prepared", download=True, allow_subset=True)
    rows = validate_manifest(manifest, True)["records"]
    assert acquired == [dataset]
    assert sum(r["split"] == "train" for r in rows) == (8 if dataset == "sun397" else 10)
    assert sum(r["split"] == "test" for r in rows) == (2 if dataset == "sun397" else 10)


def test_hf_recipe_pins_revision_and_partitions_pool(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    revision = "a" * 40
    class Provider(list):
        _fingerprint = "fixture"
    rows = Provider([{"writer_id": str(w), "character": i % 3, "image": Image.new("L", (28, 28), i)}
                     for w in range(3) for i in range(10)])
    def load_dataset(name, **kwargs):
        assert name == "flwrlabs/femnist"
        assert kwargs["revision"] == revision and kwargs["split"] == "train"
        return rows
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    manifest = prepare_femnist(tmp_path / "prepared", revision=revision)
    assert validate_manifest(manifest)["provenance"]["revision"] == revision
    partitions = femnist_partitions(manifest, 2, 42, .1, 0.)
    assert sum(len(p["train"]) for p in partitions["partitions"]) == 3
    assert sum(len(p["test"]) for p in partitions["partitions"]) == 3


def test_official_archive_import_checks_checksum_and_extracts(tmp_path, monkeypatch):
    from ae_psl import prepare_data
    import hashlib
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo("directory/file.txt")
        member.size = 7
        stream.addfile(member, io.BytesIO(b"fixture"))
    monkeypatch.setattr(prepare_data.datasets.CIFAR100, "tgz_md5", hashlib.md5(archive.read_bytes()).hexdigest())
    result = prepare_data.acquire_official("cifar100", tmp_path / "raw", archive)
    assert (tmp_path / "raw/directory/file.txt").read_text() == "fixture"
    assert len(result["archive_sha256"]) == 64
    archive.write_bytes(b"changed")
    with pytest.raises(ValueError, match="official dataset checksum"):
        prepare_data.acquire_official("cifar100", tmp_path / "other", archive)
