"""Prepare portable, checked inputs for the actual AE-PSL dataset loaders."""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
import tempfile
import tarfile
from urllib.parse import urlparse

import numpy as np
from PIL import Image
import torch
from torchvision import datasets

COUNTS = {"cifar100": 100, "food101": 101, "sun397": 397, "imagenet100": 100, "femnist": 62}
SUN397_MIRROR = ("https://huggingface.co/zhengli97/SUN397_Dataset/resolve/"
                 "99bbf6ad18ca4f01e47f8f96ca1a8b7e53cb235a/SUN397.tar.gz")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_manifest(path, check_hashes=False):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if value.get("schema") != 1 or value.get("dataset") not in COUNTS:
        raise ValueError("Unknown prepared dataset schema")
    classes, rows = value["classes"], value["records"]
    if not rows or len(set(classes)) != len(classes):
        raise ValueError("Empty records or duplicate classes")
    if len(classes) != COUNTS[value["dataset"]] and not value.get("allow_subset"):
        raise ValueError("Unexpected class count; subset manifests must say allow_subset")
    ids, files = set(), set()
    for row in rows:
        if row["id"] in ids or type(row["label"]) is not int or not 0 <= row["label"] < len(classes):
            raise ValueError("Duplicate ID or invalid label")
        ids.add(row["id"])
        file = (path.parent / row["path"]).resolve()
        if Path(row["path"]).is_absolute() or not file.is_relative_to(path.parent):
            raise ValueError("Image escapes prepared root")
        stat = file.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in files:
            raise ValueError("Duplicate image file or train/test overlap")
        files.add(identity)
        allowed = {"train", "test"} if value["dataset"] != "femnist" else {"train", "test", "pool"}
        if row["split"] not in allowed:
            raise ValueError("Invalid split")
        if value["dataset"] == "femnist" and not row.get("writer"):
            raise ValueError("Missing writer identity")
        if check_hashes:
            with Image.open(file) as image:
                image.load()
                if min(image.size) < 1:
                    raise ValueError("Empty image")
            if sha256(file) != row["sha256"]:
                raise ValueError(f"Image checksum changed: {file}")
    splits = {r["split"] for r in rows}
    if value["dataset"] == "femnist":
        if splits not in ({"pool"}, {"train", "test"}):
            raise ValueError("FEMNIST must use either pooled writers or explicit train/test splits")
    elif splits != {"train", "test"}:
        raise ValueError("Both train and test must be nonempty")
    return value


def _store_image(stage, rows, image, label, split, identity, writer=None):
    if type(label) is not int:
        label = int(label)
    # Numeric output names avoid trusting provider paths or writer IDs as paths.
    name = f"images/{len(rows):09d}.png"
    path = stage / name
    path.parent.mkdir(exist_ok=True)
    image.load()
    image.convert("L" if writer is not None else "RGB").save(path)
    row = {"id": identity, "path": name, "label": label, "split": split, "sha256": sha256(path)}
    if writer is not None:
        row["writer"] = str(writer)
    rows.append(row)


def extract_tar_checked(archive, destination):
    """Extract only regular files/directories, confined to a fresh staging root."""
    destination = Path(destination).resolve()
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if (Path(member.name).is_absolute() or not target.is_relative_to(destination)
                    or not (member.isfile() or member.isdir())):
                raise ValueError(f"Unsafe archive member: {member.name}")
        stream.extractall(destination, members=members, filter="data")


def acquire_official(dataset, root, archive=None):
    from torchvision.datasets.utils import download_url, check_md5
    constructor = {"cifar100": datasets.CIFAR100, "food101": datasets.Food101, "sun397": datasets.SUN397}[dataset]
    if dataset == "cifar100":
        url, checksum = constructor.url, constructor.tgz_md5
    elif dataset == "sun397":
        url, checksum = constructor._DATASET_URL, constructor._DATASET_MD5
    else:
        url, checksum = constructor._URL, constructor._MD5
    # Food101's canonical host supports TLS; retain its published checksum.
    url = url.replace("http://", "https://", 1)
    filename = Path(urlparse(url).path).name
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if archive is not None:
        if not check_md5(archive, checksum):
            raise ValueError("Archive does not match the official dataset checksum")
        if Path(archive).resolve() != (root / filename).resolve():
            shutil.copyfile(archive, root / filename)
    else:
        try:
            download_url(url, root, filename=filename, md5=checksum)
        except (OSError, RuntimeError):
            if dataset != "sun397":
                raise
            # This is a raw-archive mirror, never a differently split HF dataset.
            # The official checksum remains mandatory before any extraction.
            url = SUN397_MIRROR
            print("SUN397 official download unavailable; trying the checksum-checked raw archive mirror.", flush=True)
            download_url(url, root, filename=filename, md5=checksum)
    if not check_md5(root / filename, checksum):
        raise ValueError("Archive does not match the official dataset checksum")
    extract_tar_checked(root / filename, root)
    return {"url": url, "archive_md5": checksum, "archive_sha256": sha256(root / filename)}


def prepare_images(dataset, root, source=None, download=False, allow_subset=False, archive=None, raw_root=None,
                   official_source=None):
    """Import explicit train/test folders, or acquire official torchvision data."""
    if dataset not in COUNTS or dataset == "femnist":
        raise ValueError("Expected a supported image dataset")
    if sum([bool(source), bool(download), bool(archive), bool(official_source)]) != 1:
        raise ValueError("Choose exactly one folder source, official source, download or archive")
    if dataset == "imagenet100" and (download or archive):
        raise ValueError("ImageNet requires user-acquired licensed data and an explicit class subset")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    output = root / dataset
    if output.exists():
        raise FileExistsError(f"Use a fresh preparation root; preserving {output}")
    with tempfile.TemporaryDirectory(prefix=f".{dataset}-", dir=root) as temporary:
        stage = Path(temporary) / "prepared"
        stage.mkdir()
        rows, classes = [], None
        provenance = {"mode": "import", "source": str(Path(source).resolve()) if source else None}
        if source:
            source = Path(source).resolve()
            evaluation = "test" if (source / "test").is_dir() else "val"
            if not (source / "train").is_dir() or not (source / evaluation).is_dir():
                raise ValueError(f"Expected {source}/train/<class>/ and {source}/test/<class>/ (or val/<class>/)")
            if (source / "test").exists() and (source / "val").exists():
                raise ValueError("Source contains both test and val; choose one held-out split in a separate source directory")
            provenance["evaluation_source_split"] = evaluation
            seen = set()
            for split in ("train", "test"):
                data = datasets.ImageFolder(source / (evaluation if split == "test" else split))
                if classes is None:
                    classes = data.classes
                elif data.classes != classes:
                    raise ValueError("Train/test class folders must match exactly")
                for file, label in data.samples:
                    file = Path(file)
                    if not file.resolve().is_relative_to(source):
                        raise ValueError("Source image escapes import root")
                    stat = file.stat()
                    if (stat.st_dev, stat.st_ino) in seen:
                        raise ValueError("Repeated source file across splits")
                    seen.add((stat.st_dev, stat.st_ino))
                    with Image.open(file) as image:
                        _store_image(stage, rows, image, label, split, file.relative_to(source).as_posix())
        else:
            if official_source is not None:
                raw = Path(official_source).resolve()
                provenance = {"mode": "official-local", "source": str(raw)}
            else:
                raw = Path(raw_root) if raw_root is not None else Path(temporary) / "official"
                provenance = {"mode": "torchvision", **acquire_official(dataset, raw, archive)}
            provenance["torchvision"] = __import__("torchvision").__version__
            if dataset == "sun397":
                data = datasets.SUN397(raw, download=False)
                classes = data.classes
                pairs = sorted(zip(data._image_files, data._labels), key=lambda pair: pair[0].as_posix())
                data._image_files, data._labels = map(list, zip(*pairs))
                order = torch.randperm(len(data), generator=torch.Generator().manual_seed(42)).tolist()
                cut = int(.8 * len(data))
                collections = [("train", data, order[:cut]), ("test", data, order[cut:])]
                provenance["split"] = "80/20 random split; lexically ordered paths then torch seed 42"
            else:
                constructor = datasets.CIFAR100 if dataset == "cifar100" else datasets.Food101
                collections = []
                for split in ("train", "test"):
                    kwargs = {"train": split == "train"} if dataset == "cifar100" else {"split": split}
                    data = constructor(raw, download=False, **kwargs)
                    if classes is None:
                        classes = data.classes
                    elif classes != data.classes:
                        raise ValueError("Provider class mappings differ across splits")
                    collections.append((split, data, range(len(data))))
            seen = set()
            for split, data, indices in collections:
                for index in indices:
                    if official_source is not None and hasattr(data, "_image_files"):
                        original = Path(data._image_files[index])
                        if not original.resolve().is_relative_to(raw.resolve()):
                            raise ValueError("Official source image escapes import root")
                        stat = original.stat()
                        identity = (stat.st_dev, stat.st_ino)
                        if identity in seen:
                            raise ValueError("Repeated source file across splits")
                        seen.add(identity)
                    image, label = data[index]
                    _store_image(stage, rows, image, label, split, f"{split}/{index}")
        if len(classes) != COUNTS[dataset] and not allow_subset:
            raise ValueError(f"Expected {COUNTS[dataset]} classes; use --allow-subset only for a declared fixture")
        if dataset == "imagenet100":
            provenance["class_selection"] = "supplied class folders"
        write_json(stage / "manifest.json", {"schema": 1, "dataset": dataset, "classes": classes,
                   "records": rows, "allow_subset": allow_subset, "provenance": provenance})
        validate_manifest(stage / "manifest.json", True)
        stage.rename(output)
    return output / "manifest.json"


def resolve_femnist_revision():
    """Resolve main once; the full immutable commit is then used and recorded."""
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise ImportError("Install preparation dependencies: python -m pip install -r requirements.txt") from error
    revision = HfApi().dataset_info("flwrlabs/femnist", revision="main").sha
    if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision.lower()):
        raise ValueError("Hugging Face did not return an immutable 40-character FEMNIST revision")
    return revision


def prepare_femnist(root, source=None, revision=None):
    """Import LEAF JSON splits, or resolve and pin the current public HF data."""
    if source and revision:
        raise ValueError("Choose either --source for LEAF JSON or --revision for Hugging Face")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    output = root / "femnist"
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(prefix=".femnist-", dir=root) as temporary:
        stage = Path(temporary) / "prepared"
        stage.mkdir()
        rows = []
        if source:
            source = Path(source).resolve()
            input_files = []
            for split in ("train", "test"):
                files = sorted((source / split).glob("*.json"))
                if not files:
                    raise ValueError("Expected LEAF train/*.json and test/*.json")
                seen_writers = set()
                for file in files:
                    if not file.resolve().is_relative_to(source):
                        raise ValueError("LEAF file escapes source")
                    value = json.loads(file.read_text())
                    for writer in sorted(value["user_data"]):
                        if writer in seen_writers:
                            raise ValueError("A writer appears twice in the same LEAF split")
                        seen_writers.add(writer)
                        data = value["user_data"][writer]
                        if len(data["x"]) != len(data["y"]):
                            raise ValueError("LEAF images and labels have different lengths")
                        for index, (pixels, label) in enumerate(zip(data["x"], data["y"])):
                            pixels = np.asarray(pixels, dtype=np.float64)
                            if pixels.size != 784 or not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
                                raise ValueError("LEAF images must contain 784 finite values in [0,1]")
                            if type(label) is not int or not 0 <= label < 62:
                                raise ValueError("FEMNIST labels must be integers in [0,61]")
                            image = Image.fromarray(np.rint(pixels.reshape(28, 28) * 255).astype(np.uint8))
                            _store_image(stage, rows, image, label, split, f"{split}/{writer}/{index}", writer)
                    input_files.append({"path": file.relative_to(source).as_posix(), "sha256": sha256(file)})
            provenance = {"mode": "leaf-json", "source": str(source), "files": input_files,
                          "split": "provided train/test membership"}
        else:
            revision = revision or resolve_femnist_revision()
            if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision.lower()):
                raise ValueError("Supply a pinned 40-character HF commit revision")
            try:
                from datasets import load_dataset
            except ImportError as error:
                raise ImportError("Install preparation dependencies: python -m pip install -r requirements.txt") from error
            data = load_dataset("flwrlabs/femnist", split="train", revision=revision, cache_dir=str(Path(temporary) / "hf-cache"))
            for index, row in enumerate(data):
                if not 0 <= int(row["character"]) < 62:
                    raise ValueError("Invalid provider character label")
                _store_image(stage, rows, row["image"], row["character"], "pool", str(index), row["writer_id"])
            provenance = {"mode": "huggingface", "dataset": "flwrlabs/femnist", "revision": revision,
                          "fingerprint": data._fingerprint, "split": "provider train split retained as writer pool"}
        write_json(stage / "manifest.json", {"schema": 1, "dataset": "femnist", "classes": list(range(62)),
                   "records": rows, "allow_subset": False, "provenance": provenance})
        validate_manifest(stage / "manifest.json", True)
        stage.rename(output)
    return output / "manifest.json"


def legacy_main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    images = commands.add_parser("images", help="Download supported public images or import train/test class folders")
    images.add_argument("--dataset", choices=[x for x in COUNTS if x != "femnist"], required=True)
    images.add_argument("--root", type=Path, default=Path("data/datasets"))
    mode = images.add_mutually_exclusive_group(required=True)
    mode.add_argument("--download", action="store_true")
    mode.add_argument("--source", type=Path)
    mode.add_argument("--archive", type=Path, help="Previously acquired official archive; verify published checksum before extraction")
    images.add_argument("--allow-subset", action="store_true", help="Explicit small-fixture class-count exception")
    writers = commands.add_parser("femnist", help="Import LEAF JSON or download a pinned HF dataset")
    writers.add_argument("--root", type=Path, default=Path("data/datasets"))
    mode = writers.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source", type=Path)
    mode.add_argument("--revision")
    validate = commands.add_parser("validate", help="Decode images and recheck every content hash")
    validate.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "images":
        print(prepare_images(args.dataset, args.root, args.source, args.download, args.allow_subset, args.archive))
    elif args.command == "femnist":
        print(prepare_femnist(args.root, args.source, args.revision))
    else:
        value = validate_manifest(args.manifest, True)
        print(f"Validated {len(value['records'])} images for {value['dataset']}")
    return 0


def imagenet_instructions(source):
    return (f"ImageNet100 manual input required. Obtain ImageNet under its provider terms at "
            f"https://www.image-net.org/download.php and place the same chosen 100 class folders in "
            f"{source}/train/<class>/ and {source}/test/<class>/ (or {source}/val/<class>/). "
            "Then rerun python scripts/prepare_data.py --dataset imagenet100 with the same path options. "
            "The tool does not choose the 100 classes or download licensed ImageNet images.")


def prepare_dataset(dataset, root=Path("data/datasets"), raw_root=Path("data/raw"),
                    source=None, archive=None, revision=None, allow_subset=False):
    """Acquire one benchmark or validate and reuse its immutable prepared output."""
    root, raw_root = Path(root).expanduser().resolve(), Path(raw_root).expanduser().resolve()
    manifest = root / dataset / "manifest.json"
    if manifest.parent.exists():
        value = validate_manifest(manifest, True)
        if value["dataset"] != dataset:
            raise ValueError("Prepared manifest dataset differs from requested dataset")
        if value.get("allow_subset") and not allow_subset:
            raise ValueError("Existing data is a declared fixture; --allow-subset is required to reuse it")
        provenance = value["provenance"]
        if source is not None and provenance.get("source") != str(Path(source).expanduser().resolve()):
            raise ValueError("Existing prepared source differs; use a new --root to preserve both datasets")
        if revision is not None and provenance.get("revision") != revision:
            raise ValueError("Existing prepared revision differs; use a new --root")
        if archive is not None and provenance.get("archive_sha256") != sha256(archive):
            raise ValueError("Existing prepared archive differs; use a new --root")
        print(f"{dataset}: validated existing {manifest}", flush=True)
        return manifest
    local = Path(source).expanduser().resolve() if source is not None else raw_root / dataset
    print(f"{dataset}: preparing data", flush=True)
    if dataset == "femnist":
        local_source = local if source is not None or (revision is None and (
            (local / "train").exists() or (local / "test").exists())) else None
        return prepare_femnist(root, local_source, revision)
    if source is not None or (archive is None and (
            (local / "train").exists() or (local / "test").exists() or (local / "val").exists())):
        return prepare_images(dataset, root, source=local, allow_subset=allow_subset)
    if dataset == "imagenet100":
        raise FileNotFoundError(imagenet_instructions(local))
    official_marker = {"cifar100": "cifar-100-python", "food101": "food-101/meta", "sun397": "SUN397/ClassName.txt"}[dataset]
    if archive is None and (local / official_marker).exists():
        return prepare_images(dataset, root, official_source=local, allow_subset=allow_subset)
    return prepare_images(dataset, root, download=archive is None, archive=archive,
                          allow_subset=allow_subset, raw_root=local)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in {"images", "femnist", "validate"}:
        return legacy_main(argv)
    parser = argparse.ArgumentParser(description="Prepare AE-PSL datasets automatically; omit --dataset for all.")
    parser.add_argument("--dataset", choices=["all", *COUNTS], default="all")
    parser.add_argument("--root", type=Path, default=Path("data/datasets"), help="Prepared datasets; training uses --torch_data_dir")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"), help="Local source folders and downloaded official archives")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--source", type=Path, help="Single dataset: train and test/val class folders, or FEMNIST LEAF JSON")
    mode.add_argument("--archive", type=Path, help="Single public image dataset: checksum-verified official archive")
    mode.add_argument("--revision", help="FEMNIST only: optional immutable HF commit (otherwise resolve current revision)")
    parser.add_argument("--allow-subset", action="store_true", help="Explicit small-fixture class-count exception")
    args = parser.parse_args(argv)
    if args.dataset == "all" and (args.source or args.archive or args.revision):
        parser.error("--source, --archive and --revision require a single --dataset")
    if args.revision and args.dataset != "femnist":
        parser.error("--revision is only supported for --dataset femnist")
    if args.archive and args.dataset in {"imagenet100", "femnist"}:
        parser.error("--archive is only supported for cifar100, food101 and sun397")
    failed = False
    for dataset in COUNTS if args.dataset == "all" else [args.dataset]:
        local = args.raw_root / dataset
        if (args.dataset == "all" and dataset == "imagenet100" and not local.exists()
                and not (args.root / dataset).exists()):
            print(imagenet_instructions(local), flush=True)
            failed = True
            continue
        try:
            manifest = prepare_dataset(dataset, args.root, args.raw_root, args.source,
                                       args.archive, args.revision, args.allow_subset)
            print(f"{dataset}: ready at {manifest}", flush=True)
        except (OSError, ValueError, RuntimeError, ImportError, KeyError) as error:
            print(f"{dataset}: preparation failed: {error}", file=sys.stderr, flush=True)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
