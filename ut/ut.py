import sys
sys.path.append('../src')

import unittest
import numpy as np
from alternating_projections import NTTSVD
from tensor_train import TensorTrain


class TestTensorTrain(unittest.TestCase):
    def setUp(self):
        self._sizes = [10, 20, 20]
        self._ranks1 = self._ranks = [1, 10, 12, 1]
        self._ranks2 = [1, 6, 9, 1]
        self._tol = 1e-8
        self._seed = 42

    def test_TTSVD(self):
        tt = TensorTrain(sizes=self._sizes, ranks=self._ranks, seed=self._seed)
        full = TensorTrain.GetFullTensor(tt.GetCores())

        self.assertTrue(np.linalg.norm(full - TensorTrain.GetFullTensor(TensorTrain.TTSVD(full, self._ranks))) / np.linalg.norm(full) < self._tol)

    def test_DotProduct(self):
        tt1 = TensorTrain(sizes=self._sizes, ranks=self._ranks1, seed=self._seed)
        full1 = TensorTrain.GetFullTensor(tt1.GetCores())

        tt2 = TensorTrain(sizes=self._sizes, ranks=self._ranks2, seed=self._seed)
        full2 = TensorTrain.GetFullTensor(tt2.GetCores())

        self.assertTrue(abs(TensorTrain.DotProduct(tt1, tt2) - (full1 * full2).sum()) < self._tol)

    def test_Sum(self):
        tt1 = TensorTrain(sizes=self._sizes, ranks=self._ranks1, seed=self._seed)
        full1 = TensorTrain.GetFullTensor(tt1.GetCores())

        tt2 = TensorTrain(sizes=self._sizes, ranks=self._ranks2, seed=self._seed)
        full2 = TensorTrain.GetFullTensor(tt2.GetCores())

        self.assertTrue(np.linalg.norm(full1 + full2 - TensorTrain.GetFullTensor((tt1 + tt2).GetCores())) < 1e-6)

    def test_HadamardProduct(self):
        tt1 = TensorTrain(sizes=self._sizes, ranks=self._ranks1, seed=self._seed)
        full1 = TensorTrain.GetFullTensor(tt1.GetCores())

        tt2 = TensorTrain(sizes=self._sizes, ranks=self._ranks2, seed=self._seed)
        full2 = TensorTrain.GetFullTensor(tt2.GetCores())

        self.assertTrue(np.linalg.norm(full1 * full2 - TensorTrain.GetFullTensor((tt1 * tt2).GetCores())) < self._tol)

    def test_Orthogonalize(self):
        tt = TensorTrain(sizes=self._sizes, ranks=self._ranks, seed=self._seed)
        tt.Orthogonalize()

        t = tt._cores[0].squeeze(0)
        self.assertTrue(np.linalg.norm(t.T @ t - np.eye(t.shape[1])) < self._tol)
        for p in range(1, len(tt._cores) - 1):
            t = t @ tt._cores[p].reshape(tt._ranks[p], -1)
            t = t.reshape(t.shape[0] * tt._sizes[p], tt._ranks[p + 1])

            self.assertTrue(np.linalg.norm(t.T @ t - np.eye(t.shape[1])) < self._tol)

    def test_Compress(self):
        tt = TensorTrain(sizes=self._sizes, ranks=self._ranks, seed=self._seed)
        tt.Compress(self._tol)

        t = tt._cores[-1].squeeze(2)
        self.assertTrue(np.linalg.norm(t @ t.T - np.eye(t.shape[0])) < self._tol)
        for p in range(len(tt._cores) - 2, 0, -1):
            t = tt._cores[p].reshape(-1, tt._ranks[p + 1]) @ t
            t = t.reshape(tt._ranks[p], -1)

            self.assertTrue(np.linalg.norm(t @ t.T - np.eye(t.shape[0])) < self._tol)

    def test_GetMinElement(self):
        cores = []
        cores.append(np.array([[[1, 1], [2, 1]]]))
        cores.append(np.array([[[1], [1]], [[2], [1]]]))
        cores.append(np.array([[[1], [2]]]))

        tt = TensorTrain(cores=cores)
        full = TensorTrain.GetFullTensor(tt.GetCores())

        self.assertTrue(abs(tt.GetMinElement() - full.min()) < self._tol)

if __name__ == '__main__':
    unittest.main()
