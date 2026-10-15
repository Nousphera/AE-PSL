"""Prepare the exact torchvision ViT-B/32 checkpoint used by AE-PSL."""
import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import torch
import torchvision

from .prepare_data import sha256, write_json

WEIGHTS_URL = "https://download.pytorch.org/models/vit_b_32-d86f8d99.pth"
WEIGHTS_SHA256 = "d86f8d99b3c385515d1111e2a55e6f52347e8766bb4f29375eccf89fa6bbadb4"
WEIGHTS_SIZE = 352954833
WEIGHTS_NAME = "ViT_B_32_Weights.IMAGENET1K_V1"


def checkpoint_path(torch_home=None):
    """Use the same cache resolution as torchvision's pretrained constructor."""
    hub = Path(torch_home).expanduser() / "hub" if torch_home is not None else Path(torch.hub.get_dir())
    return hub.resolve() / "checkpoints" / Path(WEIGHTS_URL).name


def _validate_checkpoint(path):
    if path.stat().st_size != WEIGHTS_SIZE or sha256(path) != WEIGHTS_SHA256:
        raise ValueError(f"Invalid ViT-B/32 checkpoint size or SHA-256: {path}; preserving existing file")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not state or not all(isinstance(t, torch.Tensor) for t in state.values()):
        raise ValueError(f"Invalid checkpoint tensor state dictionary: {path}")


def _record_metadata(path):
    identity = {"schema": 1, "model": "vit_b_32", "weights": WEIGHTS_NAME,
                "url": WEIGHTS_URL, "sha256": WEIGHTS_SHA256, "bytes": WEIGHTS_SIZE}
    metadata = path.with_suffix(".metadata.json")
    if metadata.exists():
        saved = json.loads(metadata.read_text())
        if any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError(f"Checkpoint metadata differs from the pinned torchvision model: {metadata}")
    else:
        write_json(metadata, {**identity, "prepared_with": {"torch": torch.__version__, "torchvision": torchvision.__version__}})


def prepare_model(torch_home=None, source=None):
    """Validate on every run; publish complete verified downloads without overwriting."""
    output = checkpoint_path(torch_home)
    if output.exists():
        _validate_checkpoint(output)
        if source is not None:
            _validate_checkpoint(Path(source).expanduser().resolve())
        _record_metadata(output)
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ae-psl-vit-", dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        if source is not None:
            source = Path(source).expanduser().resolve()
            _validate_checkpoint(source)
            shutil.copyfile(source, staged)
        else:
            print(f"Downloading {WEIGHTS_NAME} (353 MB)", flush=True)
            torch.hub.download_url_to_file(WEIGHTS_URL, str(staged), hash_prefix=WEIGHTS_SHA256)
        _validate_checkpoint(staged)
        # Atomic create, never replace a cache entry that another process wrote.
        try:
            os.link(staged, output)
        except FileExistsError:
            _validate_checkpoint(output)
    _record_metadata(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-home", type=Path, help="Optional TORCH_HOME override; use the same TORCH_HOME when training")
    parser.add_argument("--source", type=Path, help="Optional existing official checkpoint to validate and import offline")
    args = parser.parse_args(argv)
    try:
        path = prepare_model(args.torch_home, args.source)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Model preparation failed: {error}", file=sys.stderr)
        return 1
    print(f"ViT-B/32 IMAGENET1K_V1 ready: {path}")
    if args.torch_home is not None:
        print(f"Use TORCH_HOME={args.torch_home.expanduser().resolve()} for training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
