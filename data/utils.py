import numpy as np

def one_hot(depth, indices):
    """one hot compute"""
    res = np.eye(depth, dtype=np.bool_)[indices.reshape(-1)]
    return res.reshape(list(indices.shape) + [depth])

def pad_axis(array, n, axes, constant_values=0.0):
    return np.pad(
        array,
        [(0, 0) if not i in axes else (0, n-array.shape[i]) for i in range(array.ndim)],
        mode='constant',
        constant_values=constant_values)