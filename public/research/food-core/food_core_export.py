"""Export the exact used core constants and fresh adapters for NumPy inference."""
import numpy as np
import torch
from .food_core import FoodCore


@torch.no_grad()
def arrays(bridge, body_ids):
    if type(bridge) is not FoodCore or bridge.sensor.weight.dtype != torch.float32:
        raise ValueError('A float32 food-core bridge is required')
    if bridge.core.config.neurons > 2048: raise ValueError('Dense food-core export is limited to 2,048 neurons')
    body_ids = np.asarray(body_ids)
    if (body_ids.shape != (bridge.core.config.neurons,) or body_ids.dtype.kind not in ('i', 'u')
            or np.any(body_ids < 0) or np.any(body_ids > np.iinfo(np.int64).max)
            or len(np.unique(body_ids)) != len(body_ids)):
        raise ValueError('Ordered unique int64-compatible neuron IDs required')
    core = bridge.core; recurrent, alpha, beta, gain = core.constants()
    if recurrent.is_sparse: recurrent = recurrent.to_dense()
    tensors = dict(recurrent=recurrent, alpha=alpha, beta=beta, gain=gain,
        input_weight=core.input.weight, input_bias=core.input.bias,
        pool_index=core.pool_index, pool_sizes=core.pool_sizes,
        norm_weight=core.norm.weight, norm_bias=core.norm.bias,
        sensor_weight=bridge.sensor.weight, sensor_bias=bridge.sensor.bias,
        action_weight=bridge.action.weight, action_bias=bridge.action.bias)
    result = {name: value.detach().cpu().contiguous().numpy().copy() for name, value in tensors.items()}
    result.update(norm_epsilon=np.asarray(core.norm.eps, dtype=np.float32), body_ids=body_ids.astype(np.int64))
    return result
