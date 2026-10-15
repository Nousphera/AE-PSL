"""Portable prepared-image datasets and deterministic FEMNIST writer splits."""
from functools import partial
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from ae_psl.prepare_data import sha256, validate_manifest, write_json


class PreparedImages(Dataset):
    custom_collate_fn = None

    def __init__(self, manifest, train=False, transform=None, **kwargs):
        self.manifest = Path(manifest)
        value = validate_manifest(self.manifest)
        self.dataset_name = value["dataset"]
        self.classes = value["classes"]
        self._classes = {str(name): index for index, name in enumerate(self.classes)}
        self.records = [r for r in value["records"] if r["split"] == ("train" if train else "test")]
        self.targets = [r["label"] for r in self.records]
        self.transform = transform
        self.data_processor = None
        self.source_digest = sha256(self.manifest)

    @property
    def num_classes(self):
        return len(self.classes)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(self.manifest.parent / row["path"]) as source:
            image = source.convert("RGB")
        if self.dataset_name == "cifar100":
            # The original CIFAR trainer applies its GPU-compatible transform.
            return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float(), row["label"]
        return self.transform(image) if self.transform else image, row["label"]


def prepared_factory(dataset):
    canonical = "imagenet100" if dataset == "imagenet" else dataset
    manifest = Path(os.environ["TORCH_DATA_DIR"]) / canonical / "manifest.json"
    if manifest.exists():
        if validate_manifest(manifest)["dataset"] != canonical:
            raise ValueError("Prepared manifest dataset differs from requested dataset")
        return partial(PreparedImages, manifest=manifest)
    return None


def femnist_partitions(manifest, clients, seed, fraction=.1, validation_split=0.):
    """Keep every writer; preserve explicit splits or split pooled writer samples."""
    if clients < 1 or seed < 0 or not 0 < fraction <= 1 or not 0 <= validation_split < 1:
        raise ValueError("Invalid writer partition settings")
    manifest = Path(manifest).resolve()
    value = validate_manifest(manifest)
    if value["dataset"] != "femnist":
        raise ValueError("Expected FEMNIST manifest")
    rows = value["records"]
    by_writer = {}
    for index, row in enumerate(rows):
        by_writer.setdefault(row["writer"], []).append(index)
    writers = sorted(by_writer)
    if len(writers) < clients:
        raise ValueError("Need at least one writer per client")
    signature = {"schema": 1, "manifest_sha256": sha256(manifest), "clients": clients,
                 "seed": seed, "fraction": fraction, "validation_split": validation_split}
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache = manifest.parent / "partitions" / f"{digest}.json"
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved["signature"] != signature:
            raise ValueError("Partition cache identity mismatch")
        return saved
    rng = np.random.default_rng(seed)
    groups = np.array_split(np.asarray(writers)[rng.permutation(len(writers))], clients)
    partitions = []
    for client, group in enumerate(groups):
        training, test = [], []
        for writer in group.tolist():
            ids = by_writer[writer]
            local_seed = int.from_bytes(hashlib.sha256(f"{seed}/{writer}".encode()).digest()[:8], "little")
            local_rng = np.random.default_rng(local_seed)
            if rows[ids[0]]["split"] == "pool":
                if len(ids) < 2:
                    raise ValueError(f"Writer {writer} needs at least two examples")
                sampled = local_rng.permutation(ids)[:max(2, int(len(ids) * fraction))].tolist()
                ntest = max(1, int(.2 * len(sampled)))
                test.extend(sampled[:ntest])
                training.extend(sampled[ntest:])
            else:
                for split, destination in (("train", training), ("test", test)):
                    candidates = [i for i in ids if rows[i]["split"] == split]
                    if not candidates:
                        raise ValueError(f"Writer {writer} is missing its {split} split")
                    destination.extend(local_rng.permutation(candidates)[:max(1, int(len(candidates) * fraction))].tolist())
        validation = []
        if validation_split:
            if len(training) < 2:
                raise ValueError(f"Client {client} lacks enough training examples for validation")
            shuffled = rng.permutation(training).tolist()
            nvalidation = min(len(shuffled) - 1, max(1, int(len(shuffled) * validation_split)))
            validation, training = shuffled[:nvalidation], shuffled[nvalidation:]
        partitions.append({"client": client, "writers": group.tolist(), "train": training, "val": validation, "test": test})
    result = {"signature": signature, "partitions": partitions}
    # A competing process may have produced the identical deterministic cache.
    try:
        write_json(cache, result)
    except FileExistsError:
        if json.loads(cache.read_text()) != result:
            raise ValueError("Conflicting partition cache")
    return result


class WriterImages(Dataset):
    num_classes = 62

    def __init__(self, manifest, records, indices, transform):
        self.root = Path(manifest).parent
        self.records = [records[i] for i in indices]
        self.transform = transform
        self.targets = [r["label"] for r in self.records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(self.root / row["path"]) as source:
            image = source.convert("RGB")
        return self.transform(image), row["label"]


def femnist_loaders(args, mini_batch_size, total_batch_size):
    from ae_psl.available_datasets import get_vit_mnist_transforms
    from ae_psl.utils.dataloader_utils import ClientAwareDataset, FederatedBatchSampler, federated_collate_fn
    manifest = Path(args["torch_data_dir"]) / "femnist/manifest.json"
    if not manifest.exists():
        raise FileNotFoundError("Prepare FEMNIST first: python scripts/prepare_data.py femnist --help")
    split = args["val_split"] if args["val_mode"] != "none" else 0.
    saved = femnist_partitions(manifest, args["nr_of_clients"], args["random_seed"],
                               args.get("femnist_fraction", .1), split)
    value = validate_manifest(manifest)
    training, ownership, validation, testing = [], {}, {}, {}
    offset = 0
    for part in saved["partitions"]:
        cid = part["client"]
        train = WriterImages(manifest, value["records"], part["train"], get_vit_mnist_transforms())
        training.append(train)
        ownership[cid] = list(range(offset, offset + len(train)))
        offset += len(train)
        for key, destination in (("val", validation), ("test", testing)):
            if part[key]:
                data = WriterImages(manifest, value["records"], part[key], get_vit_mnist_transforms())
                destination[cid] = DataLoader(data, batch_size=total_batch_size, shuffle=False,
                                              num_workers=args.get("test_num_workers", 0))
    combined = ConcatDataset(training)
    loader = DataLoader(ClientAwareDataset(combined, ownership),
                        batch_sampler=FederatedBatchSampler(ownership, mini_batch_size, shuffle=True, seed=args["random_seed"]),
                        collate_fn=federated_collate_fn, num_workers=args["num_workers_finetuning"])
    args["femnist_preparation"] = saved["signature"]
    args["femnist_writer_partitions"] = saved["partitions"]
    return loader, validation, testing


class PreparedAlignmentData:
    """The selected writer training data, excluding all downstream validation data."""

    def __init__(self, training_loader, testing_loaders, signature):
        self.train_ds = training_loader.dataset.base_dataset
        self.test_ds = ConcatDataset([loader.dataset for loader in testing_loaders.values()])
        self.source_digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()

    def load_test_set(self):
        return self.test_ds

    def get_collate_fn(self):
        return None
