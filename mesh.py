'''
Mesh operation functions for this project.

Author: Gerald Baulig
'''

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial import Delaunay


def face_normals(T, normalize=True):
    fN = np.cross(T[:,1] - T[:,0], T[:,2] - T[:,0])
    if normalize:
        return fN / np.linalg.norm(fN, axis=1)[:, None]
    else:
        return fN


def edge_normals(fN, Ti_flat, normalize=True):
    fN = fN.repeat(3, axis=0)
    eN = np.zeros((Ti_flat.max()+1, 3))
    for fn, i in zip(fN, Ti_flat):
        eN[i] += fn
    if normalize:
        return eN / np.linalg.norm(eN, axis=1)[:, None]
    else:
        return eN


def nn_point2point(X, Y):
    return KDTree(Y).query(X)


def nn_point2line(X, Xi, P):
    dist = np.zeros(P.shape)
    nn = -np.ones(P.shape[0])
    A = X[Xi[:,0]]
    B = X[Xi[:,1]]
    AB = B - A
    AB_norm = np.linalg.norm(AB, axis=1)[:, None]
    ABi = list(range(AB.shape[0]))
    for i, p in enumerate(P):
        Ap = p - A
        Bp = p - B
        a = np.sum(AB * Ap, axis=1)
        b = np.sum(-AB * Bp, axis=1)
        m = (a * b) > 0
        if any(m):
            L = np.cross(Ap[m], AB[m]) / AB_norm[m]
            Larg = np.argmin(L)
            print(m)
            nn[i] = ABi[m][Larg]
            dist[i] = L[Larg]
    
    m = nn > 0
    mnn, mdist = KDTree(X[m]).query(P[m])
    nn[m] = ABi[m][mnn]
    dist[m] = mdist[nn[m]]
    return nn, dist


def polarize(X, scale=(10,10)):
    P = X.copy()
    P[:,0] = np.arccos(X[:,0] / np.linalg.norm(X[:,(0,1)], axis=1)) * (1*(X[:,1] >= 0) - (X[:,1] < 0)) * scale[0]
    P[:,1] = np.linalg.norm(X[:,:2], axis=1)
    P[:,2] = np.arcsin(P[:,2] / P[:,1]) * scale[1]
    return P


#TEST
if __name__ == '__main__':
    X = np.random.rand(20,3)
    P = np.random.rand(15,3)
    Xi = np.array((range(X.shape[0]-1), range(1,X.shape[0]))).T
    nn, dist = nn_point2line(X, Xi, P)
    print(nn)
    print(dist)
    