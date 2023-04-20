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

    def GetMinElement(self, algorithm='power', *args, **kwargs):
        if algorithm == 'als':
            return self._Als(*args, **kwargs)
        if algorithm == 'power':
            return self._PowerMethod(*args, **kwargs)
        assert(False, 'Unknown algorithm')

    def _Als(self, itersNum, tol=1e-8): # ALS
        x = TensorTrain(cores=[core.copy() for core in self._cores])
        prevMinElement = None
        for k in range(itersNum):
            x.Compress(tol)
            for p in range(len(x._cores)):
                # A * x = \lambda * x
                m = x._ranks[p] * x._sizes[p]
                n = x._ranks[p + 1]
                x._cores[p][ :, :, :] = 0
                # computing A
                a = np.empty((m * n, m * n))
                for i in range(m * n):
                    x._cores[p][x._GetPos(p, i)] = 1
                    tmp = self * x
                    x._cores[p][x._GetPos(p, i)] = 0
                    for j in range(i, m * n):
                        x._cores[p][x._GetPos(p, j)] = 1
                        a[i, j] = TensorTrain.DotProduct(tmp, x)
                        x._cores[p][x._GetPos(p, j)] = 0
                # computing the eigenvalues and eigenvectors
                w, v = np.linalg.eigh(a, UPLO='U')
                minElement = w[0]
                # orthogonalization
                q, r = np.linalg.qr(v[ :, 0].reshape(m, n))
                if m < n:
                    x._ranks[p + 1]
                # update core
                x._cores[p] = q.reshape(x._ranks[p], x._sizes[p], x._ranks[p + 1])
        return minElement

    def _PowerMethod(self, stage1, stage2, tol=1e-8, debug=False):
        x = TensorTrain(cores=[core.copy() for core in self._cores])
        # stage 1
        for _ in range(stage1):
            newX = self * x
            newX.Compress(tol)
            newX.Normalize()
            x = newX
        maxElement = TensorTrain.DotProduct(self * x, x)
        if debug:
            fullTensor = TensorTrain.GetFullTensor(self._cores)
            print('maxElement (true/found):', fullTensor.max(), maxElement)
            print('x._ranks', x._ranks)

        rank1Tensor = TensorTrain.GetRank1Tensor(self._sizes)
        rank1Tensor._cores[0] *= -maxElement
        newTensor = self + rank1Tensor

        # stage 2
        x = TensorTrain(cores=[core.copy() for core in newTensor._cores])
        for _ in range(stage2):
            newX = newTensor * x
            newX.Compress(tol)
            newX.Normalize()
            x = newX
        if debug:
            print('minElement (true/found):', fullTensor.min(), TensorTrain.DotProduct(newTensor * x, x) + maxElement)
            print('x._ranks', x._ranks)

        return TensorTrain.DotProduct(newTensor * x, x) + maxElement

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
                self._ranks[p + 1] = m

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
            u *= s

            rank = min(m, n)
            if maxRank is not None:
                rank = min(rank, maxRank)
            curError = sum([sigma * sigma for sigma in s[rank : ]])
            while rank > 0 and curError + (s[rank - 1] * s[rank - 1]) * p < maxError:
                rank -= 1
                curError += s[rank] * s[rank]

            self._cores[p] = vh[ : rank, : ].reshape(rank, self._sizes[p], self._ranks[p + 1])
            self._cores[p - 1] = self._cores[p - 1].reshape(-1, self._ranks[p]) @ u[ :, : rank]
            self._cores[p - 1] = self._cores[p - 1].reshape(self._ranks[p - 1], self._sizes[p - 1], rank)
            self._ranks[p] = rank

        self._isLeftToRight = False
        self._isRightToLeft = True

    def __mul__(self, other): # Hadamard product
        assert self._sizes == other._sizes

        cores = []
        for p in range(len(self._sizes)):
            core = np.empty((self._ranks[p] * other._ranks[p], self._sizes[p], self._ranks[p + 1] * other._ranks[p + 1]))
            for i in range(self._sizes[p]):
                core[ :, i, : ] = np.kron(self._cores[p][ :, i, : ], other._cores[p][ :, i, : ])
            cores.append(core)

        return TensorTrain(cores=cores)

    def __add__(self, other):
        assert self._sizes == other._sizes

        cores = []
        cores.append(np.block([self._cores[0][0], other._cores[0][0]])[np.newaxis, :, : ])
        for p in range(1, len(self._sizes) - 1):
            a = np.concatenate((self._cores[p], np.zeros((self._ranks[p], self._sizes[p], other._ranks[p + 1]))), axis=2)
            b = np.concatenate((np.zeros((other._ranks[p], other._sizes[p], self._ranks[p + 1])), other._cores[p]), axis=2)
            cores.append(np.concatenate((a, b), axis=0))
        cores.append(np.vstack([self._cores[-1][ :, :, 0], other._cores[-1][ :, :, 0]])[ :, :, np.newaxis])

        return TensorTrain(cores=cores)

    def Norm(self):
        return np.sqrt(TensorTrain.DotProduct(self, self))

    def Normalize(self):
        self._cores[0] /= self.Norm()

    def _GetPos(self, p, i):
        return (i // (self._sizes[p] * self._ranks[p + 1]) % self._ranks[p], i // self._ranks[p + 1] % self._sizes[p], i % self._ranks[p + 1])

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

    @staticmethod
    def GetRank1Tensor(sizes):
        return TensorTrain(cores=[np.ones((1, size, 1)) for size in sizes])

    def GetCores(self, p=None):
        if p is not None:
            return self._cores[p]
        return self._cores

    def GetSizes(self):
        return self._sizes

    def GetRanks(self):
        return self._ranks
