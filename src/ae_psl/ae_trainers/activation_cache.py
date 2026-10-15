"""Bounded, atomic frozen-feature caches used across AE architecture sweeps."""
import hashlib
import json
import os
import tempfile
import shutil
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from tqdm import tqdm


EXTRACTION_BATCH_SIZE = 64


def _open_cache(path):
    try:
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        if not isinstance(array, np.memmap) or array.ndim != 3 or array.dtype != np.float32:
            raise ValueError('expected a float32 sample/token/feature array')
        if any(size < 1 for size in array.shape):
            raise ValueError('empty feature dimensions')
        if Path(path).stat().st_size != array.offset + array.nbytes:
            raise ValueError('file length differs from its declared shape')
        return array
    except (OSError, ValueError, EOFError) as error:
        raise ValueError(f'Invalid activation cache {path}: {error}') from error


class ActivationDataset(Dataset):
    """Map immutable features lazily in each process and copy only one sample."""
    def __init__(self, path, expected_count=None):
        self.path = str(Path(path).resolve())
        array = _open_cache(self.path)
        self.shape = tuple(array.shape)
        array._mmap.close()
        if expected_count is not None and self.shape[0] != expected_count:
            raise ValueError('Activation cache sample count differs from the source dataset')
        self._mapped = None

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        if self._mapped is None:
            self._mapped = _open_cache(self.path)
        # A writable independent sample avoids read-only numpy aliasing in torch.
        return (torch.from_numpy(np.array(self._mapped[index], copy=True)),)

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_mapped'] = None  # Spawned workers reopen instead of pickling data.
        return state


def write_activation_cache(base_model, split_layer, dataloader, device, cache_path,
                           raw_cifar=False, seed=42):
    """Publish only a complete cache, retaining at most a feature batch in RAM."""
    path = Path(cache_path)
    count = len(dataloader.dataset)
    if count < 1:
        raise ValueError('Activation extraction requires samples')
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    os.close(descriptor)
    mapped = None
    written = 0
    base_model.eval()
    try:
        # Transforms and iteration must not depend on AE initialization or advance
        # the following GA optimizer's random stream when this cache is absent.
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.random.default_generator.manual_seed(seed)
            for images, _ in tqdm(dataloader, desc='Extracting'):
                images = images.to(device)
                if raw_cifar:
                    from ae_psl.trainers.implementations.classification.sasl_trainer import get_gpu_vit_transforms
                    images = get_gpu_vit_transforms(train=False)(images / 255.)
                features = base_model.retrieve_split_layer_activations(images, split_layer).detach().cpu()
                if features.dtype != torch.float32 or features.ndim != 3 or not torch.isfinite(features).all():
                    raise ValueError('Activation cache requires finite float32 sample/token/feature batches')
                if len(features) != len(images) or written + len(features) > count:
                    raise ValueError('Feature batch and source samples disagree')
                if mapped is None:
                    required = count * math.prod(features.shape[1:]) * features.element_size() + 4096
                    available = shutil.disk_usage(path.parent).free
                    if available < required:
                        raise OSError(f'Insufficient disk space for activation cache {path}: '
                                      f'need {required / 2**30:.2f} GiB, have {available / 2**30:.2f} GiB')
                    mapped = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32,
                        shape=(count, *features.shape[1:]))
                if tuple(features.shape[1:]) != tuple(mapped.shape[1:]):
                    raise ValueError('Feature dimensions changed during extraction')
                mapped[written:written + len(features)] = features.numpy()
                written += len(features)
                mapped.flush()
                del features
        if written != count or mapped is None:
            raise ValueError(f'Activation extraction wrote {written} of {count} source samples')
        mapped.flush()
        mapped._mmap.close()
        mapped = None
        with open(temporary, 'rb') as stream:
            os.fsync(stream.fileno())
        # Validation occurs before the previous complete cache is replaced.
        ActivationDataset(temporary, expected_count=count)
        os.replace(temporary, path)
        return ActivationDataset(path, expected_count=count)
    finally:
        if mapped is not None:
            mapped._mmap.close()
        if os.path.exists(temporary):
            os.unlink(temporary)


def feature_extractor_fingerprint(model, split_layer):
    """Fingerprint only parameters/buffers used before the split; exclude heads."""
    if hasattr(model, 'vit'):
        vit = model.vit
        state = {'class_token': vit.class_token, 'position_embedding': vit.encoder.pos_embedding}
        state.update({f'projection.{key}': value for key, value in vit.conv_proj.state_dict().items()})
        state.update({f'blocks.{key}': value for key, value in vit.encoder.layers[:split_layer].state_dict().items()})
    else:
        state = model.state_dict()
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _dataset_identity(dataset):
    identity = {'type': type(dataset).__qualname__, 'samples': len(dataset)}
    if isinstance(dataset, Subset):
        indices = np.asarray(dataset.indices, dtype=np.int64)
        identity.update(indices=hashlib.sha256(indices.tobytes()).hexdigest(),
                        dataset=_dataset_identity(dataset.dataset))
    elif isinstance(dataset, ConcatDataset):
        identity['datasets'] = [_dataset_identity(child) for child in dataset.datasets]
    else:
        identity['source_digest'] = getattr(dataset, 'source_digest', None)
        identity['transform'] = repr(getattr(dataset, 'transform', None))
    return identity


def feature_cache_signature(args, base_model, train_dataset, test_dataset, source_digest):
    """Feature identity excludes AE architecture, optimizer, epochs and head."""
    return {
        'feature_cache_schema': 1,
        'model': args['model'], 'split_layer': args['split_layer'],
        'prefix_sha256': feature_extractor_fingerprint(base_model, args['split_layer']),
        'dataset': args['ae_pretrain_dataset'],
        'dataset_root': os.path.abspath(os.environ.get('TORCH_DATA_DIR', 'data/datasets')),
        'prepared_manifest_sha256': source_digest,
        'fraction': args['ae_pretrain_dataset_fraction'],
        'small_test_run': args['small_test_run'],
        'seed': args['random_seed'],
        'extraction_workers': args['num_workers_activation_loading'],
        'extraction_batch_size': EXTRACTION_BATCH_SIZE,
        'cifar_preprocessing': 'resize224-imagenet-normalization-v1',
        'train': _dataset_identity(train_dataset), 'test': _dataset_identity(test_dataset),
    }


def feature_cache_name(signature):
    serialized = json.dumps(signature, sort_keys=True, separators=(',', ':')).encode()
    return 'features-' + hashlib.sha256(serialized).hexdigest()
