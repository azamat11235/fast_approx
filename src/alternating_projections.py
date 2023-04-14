import numpy as np
from tensor_train import TensorTrain
from truncatedSvd import TruncatedSvd


def NLRT(tensor, ranks, truncatedSvd, itersNum):
    '''https://arxiv.org/abs/2007.14137v2'''
    m = len(tensor.shape)
    X = [tensor.copy()] * m
    for _ in range(itersNum):
        # projection onto \Omega_1
        for Xi in X:
            Xi[Xi < 0] = 0
        Yi = sum(X) / m
        # projection onto \Omega_2
        for i in range(m):
            u, vh = truncatedSvd(Unfold(Yi, i), ranks[i])
            X[i] = Fold(u @ vh, i, tensor.shape)
    return X

def NTTSVD(tensor, ranks, truncatedSvd, itersNum):
    '''https://arxiv.org/abs/2209.02060'''
    tensor = tensor.copy()
    cores = TensorTrain.TTSVD(tensor, ranks, truncatedSvd)
    for _ in range(itersNum):
        tensor = TensorTrain.GetFullTensor(cores)
        tensor[tensor < 0] = 0
        cores = TensorTrain.TTSVD(tensor, ranks, truncatedSvd)
    return cores
