'''
Mesh operation functions for this project.

Author: Gerald Baulig
'''

import numpy as np
from scipy.spatial import KDTree


def magnitude(X, sqrt=False):
    if len(X.shape) == 1:
        m = np.sum(X**2)
    else:
        m = np.sum(X**2, axis=-1)[:,None]
    return np.sqrt(m) if sqrt else m


def norm(X, mgni=False):
    if len(X.shape) == 1:
        m = np.linalg.norm(X)
    else:
        m = np.linalg.norm(X, axis=-1)[:,None]
    n = X / m
    if mgni:
        return n, m
    else:
        return n


def face_normals(T, normalize=True, magnitude=False):
    fN = np.cross(T[:,1] - T[:,0], T[:,2] - T[:,0])
    if normalize:
        return norm(fN, magnitude)
    else:
        return fN


def edge_normals(fN, Ti_flat, normalize=True, magnitude=False):
    fN = fN.repeat(3, axis=0)
    eN = np.zeros((Ti_flat.max()+1, 3))
    for fn, i in zip(fN, Ti_flat):
        eN[i] += fn
    if normalize:
        return norm(eN, magnitude)
    else:
        return eN


def nn_point2point(X, P):
    return KDTree(X).query(P)


def nn_point2line(X, Xi, P):
    total = time_delta(time())
    delta = time_delta(time())
    nn = -np.ones((P.shape[0],2), dtype=int)
    dist, nn[:,0] = nn_point2point(X, P)
    print("KDTree", next(delta))
    
    mp = X[nn[:,0]]
    A = X[Xi[:,0]]
    B = X[Xi[:,1]]
    AB, ABn = norm(B - A, True)
    print("AB normals", next(delta))
    for i, p in enumerate(P):
        Ap = p - A
        Bp = p - B
        a = np.sum(AB * Ap, axis=1)
        b = np.sum(-AB * Bp, axis=1)
        m = (a * b) > 0
        if any(m):
            n, L = norm(AB[m] * a[m][:,None] + A[m] - p, True)
            Larg = np.argmin(L)
            Lmin = L[Larg]
            if Lmin < dist[i]:
                nn[i] = Xi[m][Larg]
                dist[i] = Lmin
                mp[i] = p + n[Larg] * Lmin
    print("Total", next(total)) 
    return dist, mp, nn


def polarize(X, scale=(10,10)):
    P = X.copy()
    P[:,0] = np.arccos(X[:,0] / np.linalg.norm(X[:,(0,1)], axis=1)) * (1*(X[:,1] >= 0) - (X[:,1] < 0)) * scale[0]
    P[:,1] = np.linalg.norm(X[:,:2], axis=1)
    P[:,2] = np.arcsin(P[:,2] / P[:,1]) * scale[1]
    return P


###TEST
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from utils import *
    
    np.random.seed(10)
    X = np.random.randn(10000,3)
    P = np.random.randn(10000,3)
    Xi = np.array((range(X.shape[0]-1), range(1,X.shape[0]))).T
    
    print("Brute force")
    dist, mp, nn = nn_point2line(X, Xi, P)
    print("Mean loss:", dist.mean())
    mp -= P
    
    
    fig = plt.figure()
    ax = fig.add_subplot((111), projection='3d')
    seg = np.hstack((X[Xi[:,0]], X[Xi[:,1]]-X[Xi[:,0]]))
    x, y, z, u, v, w = zip(*seg)
    
    ax.quiver(x, y, z, u, v, w)
    ax.scatter(P[:,0],P[:,1],P[:,2], color='r')
    ax.quiver(P[:,0],P[:,1],P[:,2],mp[:,0],mp[:,1],mp[:,2], color='g')
    
    ax.set_xlim3d(-3, 3)
    ax.set_ylim3d(-3, 3)
    ax.set_zlim3d(-3, 3)
    plt.show()