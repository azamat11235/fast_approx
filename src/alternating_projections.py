import numpy as np
import truncated_svd
import utils
from tensor_train import TensorTrain


def NTTSVD(tensor, ranks, itersNum, truncatedSvd=truncated_svd.SVDr):
    '''https://arxiv.org/abs/2209.02060'''
    tensor = tensor.copy()
    cores = TensorTrain.TTSVD(tensor, ranks, truncatedSvd)
    for _ in range(itersNum):
        tensor = TensorTrain.GetFullTensor(cores)
        tensor[tensor < 0] = 0
        cores = TensorTrain.TTSVD(tensor, ranks, truncatedSvd)
    return cores
