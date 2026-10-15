"""Local byte transport reference. No pickle, shared tensors, or autograd graphs.

This is an in-process serialized boundary, not a network deployment or privacy
mechanism. Logit gradients are necessary; scalar losses cannot recover them.
"""
import json
import struct

import numpy as np
import torch


def transmit(tensor: torch.Tensor, *, requires_grad: bool = False):
    """Return an independent tensor and the exact encoded message byte count."""
    dtypes = {torch.float32: '<f4', torch.int64: '<i8'}
    if tensor.dtype not in dtypes:
        raise ValueError("Reference transport supports float32 values and int64 indices")
    dtype = dtypes[tensor.dtype]
    array = tensor.detach().cpu().contiguous().numpy().astype(dtype, copy=False)
    header = json.dumps({'dtype': dtype, 'shape': list(array.shape)}, separators=(',', ':')).encode()
    payload = struct.pack('!I', len(header)) + header + array.tobytes()
    length = struct.unpack('!I', payload[:4])[0]
    metadata = json.loads(payload[4:4+length])
    restored = np.frombuffer(payload[4+length:], dtype=metadata['dtype']).reshape(metadata['shape']).copy()
    result = torch.from_numpy(restored).to(tensor.device).requires_grad_(requires_grad)
    return result, len(payload)
