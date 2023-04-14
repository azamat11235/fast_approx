import numpy as np


def Unfold(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))

def Fold(unfoldedTensor, mode, shape):
    shape = list(shape)
    shape.insert(0, shape.pop(mode))
    return np.moveaxis(np.reshape(unfoldedTensor, shape), 0, mode)

def ModeProduct(tensor, matrix, mode):
    if tensor.shape[mode] != matrix.shape[0]:
        raise RuntimeError(f'tensor.shape[{mode}] != matrix.shape[0]')
    newShape = list(tensor.shape)
    newShape[mode] = matrix.shape[1]
    return Fold(matrix.T @ Unfold(tensor, mode), mode, newShape)
