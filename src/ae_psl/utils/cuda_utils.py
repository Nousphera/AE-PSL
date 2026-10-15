"""Select a requested accelerator without silently changing the experiment."""

import torch


def get_device(arguments):
    requested = arguments.get('device', 'auto')
    gpu_id = arguments.get('gpu_id')
    if requested == 'auto' and gpu_id is not None:
        requested = 'mps' if gpu_id == 'mps' else 'cuda'
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable; select --device cpu or mps.')
        if gpu_id is not None:
            try:
                index = int(gpu_id)
            except (ValueError, TypeError) as error:
                raise ValueError('--gpu_id must be a CUDA device index') from error
            if not 0 <= index < torch.cuda.device_count():
                raise ValueError('--gpu_id is outside the available CUDA devices')
            return torch.device('cuda', index)
    elif requested == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS was requested but is unavailable; select --device cpu.')
    elif requested != 'cpu':
        raise ValueError(f'Unsupported device: {requested}')
    return torch.device(requested)
