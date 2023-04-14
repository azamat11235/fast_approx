import numpy as np
import scipy as sp


def TestMatrix(m, n, distribution='rademacher'):
    if distribution == 'normal':
        res = np.random.normal(size=(m, n))
    elif distribution == 'rademacher':
        res = np.random.choice([-1, 1], size=(m, n))
    else:
        raise TypeError('Invalid arguments')
    return res

def SVDr(a, r):
    u, s, vh = sp.linalg.svd(a, full_matrices=False)
    return u[:, :r], s[:r], vh[:r, :]

def HMT(X, rank, p, k, distribution='rademacher'):
    n = X.shape[1]

    Psi = TestMatrix(n, k, distribution)
    Z1 = X @ Psi
    Q, _ = np.linalg.qr(Z1)
    for _ in range(p):
        Z2 = Q.T @ X
        Q, _ = np.linalg.qr(Z2.T)
        Z1 = X @ Q
        Q, _ = np.linalg.qr(Z1)
    Z2 = Q.T @ X
    Ur, Sr, Vhr = SVDr(Z2, rank)
    Ur = Q @ Ur

    return Ur, Sr, Vhr

def Tropp(X, rank, k, l, distribution='rademacher'):
    m, n = X.shape

    Psi = TestMatrix(n, k, distribution)
    Phi = TestMatrix(l, m, distribution)
    Z = X @ Psi
    Q, R = np.linalg.qr(Z)
    W = Phi @ Q
    P, T = np.linalg.qr(W)
    G = np.linalg.inv(T) @ P.T @ Phi @ X
    Ur, Sr, Vhr = SVDr(G, rank)
    Ur = Q @ Ur

    return Ur, Sr, Vhr

def GN(X, rank, l, distribution='rademacher'):
    m, n = X.shape

    Psi = TestMatrix(n, rank, distribution)
    Phi = TestMatrix(l, m, distribution)
    Z = X @ Psi
    W = Phi @ Z
    Q, R = np.linalg.qr(W)
    V = (Phi @ X).T @ Q
    U = Z @ np.linalg.inv(R)

    return U, V.T
