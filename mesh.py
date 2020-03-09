'''
Mesh operation functions for this project.

Author: Gerald Baulig
'''

import numpy as np
from scipy.statial import KDTree
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
    Dnn, Di = KDTree(X).query(P)
    AB = X[Xi[:,1]] - X[Xi[:,0]]
    
    
    
    AB_norm = np.linalg.norm(AB, axis=1)[:, None]
    D = np.cross(AP, AB) / AB_norm
    return D


def polarize(X, scale=(10,10)):
    P = X.copy()
    P[:,0] = np.arccos(X[:,0] / np.linalg.norm(X[:,(0,1)], axis=1)) * (1*(X[:,1] >= 0) - (X[:,1] < 0)) * scale[0]
    P[:,1] = np.linalg.norm(X[:,:2], axis=1)
    P[:,2] = np.arcsin(P[:,2] / P[:,1]) * scale[1]
    return P