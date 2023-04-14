import numpy as np
import scipy as sp
import utils
import truncated_svd


class TensorTrain:
    def __init__(self, cores=None, sizes=None, ranks=None, seed=None):
        if cores is not None:
            assert all(len(core.shape) == 3 for core in cores)
            self._cores = cores
            sizes = [core.shape[1] for core in cores]
            ranks = [1] + [core.shape[0] for core in cores[1 : ]] + [1]
        else:
            # Fill the cores with random elements
            assert (sizes is not None) and (ranks is not None)
            assert len(sizes) >= 3
            assert len(sizes) + 1 == len(ranks)
            assert (ranks[0] == 1) and (ranks[-1] == 1)
            rng = np.random.RandomState(seed)
            self._cores = []
            for i in range(len(sizes)):
                self._cores.append(rng.rand(ranks[i], sizes[i], ranks[i + 1]))

        self._sizes = sizes
        self._ranks = ranks
        self._isLeftToRight = False
        self._isRightToLeft = False

    def GetMinElement(self): # ALS
        pass

    def Orthogonalize(self):
        # left to right
        if self._isLeftToRight:
            return

        for p in range(len(self._cores) - 1):
            m = self._ranks[p] * self._sizes[p]
            n = self._ranks[p + 1]
            q, r = np.linalg.qr(self._cores[p].reshape(m, n))
            self._cores[p] = q.reshape(self._ranks[p], self._sizes[p], min(m, n))
            self._cores[p + 1] = self._cores[p + 1].reshape(self._ranks[p + 1], -1)
            self._cores[p + 1] = (r @ self._cores[p + 1]).reshape(min(m, n), self._sizes[p + 1], self._ranks[p + 2])
            if m < n:
                self._ranks[p + 1]

        self._isLeftToRight = True

    def Compress(self, tol, maxRank=None):
        # right to left
        if not self._isLeftToRight:
            self.Orthogonalize()

        maxError = tol * self.Norm()
        error = 0
        for p in range(len(self._cores) - 1, 0, -1):
            m = self._ranks[p]
            n = self._sizes[p] * self._ranks[p + 1]
            u, s, vh = np.linalg.svd(self._cores[p].reshape(m, n), full_matrices=False)
            u = (u.T * s).T

            rank = min(m, n)
            if maxRank is not None:
                rank = min(rank, maxRank)
            curError = [sigma * sigma for sigma in s[rank : ]]
            while rank > 0 and curError + (s[rank - 1] * s[rank - 1]) * p < maxError:
                rank -= 1
                curError += s[rank] * s[rank]

            self._ranks[p] = rank
            self._cores[p] = vh[ : rank, : ].reshape(self._ranks[p], self._sizes[p], self._ranks[p + 1])
            self._cores[p - 1] = self._cores[p - 1].reshape(-1, self._ranks[p]) @ u[ :, : rank]
            self._cores[p - 1] = self._cores[p - 1].reshape(self._ranks[p - 1], self._sizes[p - 1], self._ranks[p])

        self._isLeftToRight = False
        self._isRightToLeft = True

    def __mul__(self, other): # Hadamard product
        assert self._sizes == other._sizes

        cores = []
        for p in range(len(self._sizes)):
            core = np.empty((self._ranks[p] * other._ranks[p], self._sizes[p], self._ranks[p + 1] * other._ranks[p + 1]))
            for i in range(self._cores[p].shape[1]):
                core[ :, i, : ] = np.kron(self._cores[p][ :, i, : ], other._cores[p][ :, i, : ])
            cores.append(core)

        return TensorTrain(cores=cores)

    def __add__(self, other):
        assert self._sizes == other._sizes

        cores = []
        cores.append(np.block([self._cores[0][0], other._cores[0][0]])[np.newaxis, :, : ])
        for p in range(1, len(self._sizes) - 1):
            a = np.concatenate((self._cores[p], np.zeros_like(self._cores[p])), axis=2)
            b = np.concatenate((np.zeros_like(other._cores[p]), other._cores[p]), axis=2)
            cores.append(np.concatenate((a, b), axis=0))
        cores.append(np.vstack([self._cores[-1][ :, :, 0], other._cores[-1][ :, :, 0]])[ :, :, np.newaxis])

        return TensorTrain(cores=cores)

    def Norm(self):
        return np.sqrt(TensorTrain.DotProduct(self, self))

    @staticmethod
    def DotProduct(tt1, tt2):
        assert tt1._sizes == tt2._sizes
        tmp = tt1._cores[0]
        for p in range(len(tt1._cores) - 1):
            tmp = tmp.reshape(tt2._ranks[p] * tt2._sizes[p], tt1._ranks[p + 1]).T
            tmp = tmp @ tt2._cores[p].reshape(tt2._sizes[p] * tt2._ranks[p], tt2._ranks[p + 1])
            tmp = tmp.T @ tt1._cores[p + 1].reshape(tt1._ranks[p + 1], tt1._ranks[p + 2] * tt1._sizes[p + 1])
        return (tmp * tt2._cores[-1].squeeze(2)).sum()

    @staticmethod
    def TTSVD(tensor, ranks, svdr=truncated_svd.SVDr):
        assert (ranks[0] == 1) and (ranks[-1] == 1)
        sizes = np.array(tensor.shape)
        cores = []
        for p in range(0, len(tensor.shape) - 1):
            u, s, vh = svdr(tensor.reshape(ranks[p] * sizes[p], np.prod(sizes[p + 1 : ])), ranks[p + 1])
            u = u.reshape(ranks[p], sizes[p], min(ranks[p + 1], vh.shape[0]))
            cores.append(u.reshape(ranks[p], sizes[p], min(ranks[p + 1], vh.shape[0])))
            tensor = (s * vh.T).T
        cores.append(tensor[ :, :, np.newaxis])
        return cores

    @staticmethod
    def GetFullTensor(cores):
        assert len(cores) >= 3
        fullTensor = utils.ModeProduct(cores[1], cores[0].squeeze(0).T, 0)
        for p in range(2, len(cores)):
            newShape = list(fullTensor.shape[ : -1]) + list(cores[p].shape[1 : ])
            fullTensor = utils.ModeProduct(fullTensor, utils.Unfold(cores[p], 0), p).reshape(newShape)
        return fullTensor.squeeze(-1)

    def GetCores(self):
        return self._cores

    def GetSizes(self):
        return self._sizes

    def GetRanks(self):
        return self._ranks
