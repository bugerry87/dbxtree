'''
Spatial operation functions for 3D.

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


def quantirize(P, m=1):
    k = P[0]
    p0 = P[1]
    p0k = p0 - k
    p0km = magnitude(p0k)
    mag = p0km
    mask = np.zeros(P.shape[0], dtype=bool)
    m = m**2
    
    for i, p1 in enumerate(P[2:]):
        pp = p1 - p0
        ppm = magnitude(pp)
        mag += ppm
        
        p1k = p1 - k
        p1km = magnitude(p1k)
        dot = np.dot(p0k, p1k)**2 / (p0km * p1km) 
        
        if dot < 1 - np.exp(-mag/m)**4:
            #new keypoint detected
            k = p0
            p0 = p1
            p0k = pp
            p0km = ppm
            mag = ppm
            mask[i] = True
        else:
            #update
            p0 = p1
            p0k = p1k
            p0km = p1km
    return P[mask]


def nn_point2point(X, P):
    return KDTree(X).query(P)


def nn_point2line(X, Xi, P):
    nn = -np.ones((P.shape[0],2), dtype=int)
    dist, nn[:,0] = nn_point2point(X, P)
    dist = dist**2
    mp = X[nn[:,0]]
    A = X[Xi[:,0]]
    B = X[Xi[:,1]]
    AB = B - A
    ABm = magnitude(AB)
    for i, p in enumerate(P):
        Ap = p - A
        a = np.sum(AB * Ap, axis=1) / ABm.flatten()
        m = ((a > 0) * (a < 1)).astype(bool)
        if any(m):
            ap = AB[m] * a[m][:,None] - Ap[m]
            L = magnitude(ap)
            Larg = np.argmin(L)
            Lmin = L[Larg]
            if Lmin < dist[i]:
                nn[i] = Xi[m][Larg]
                dist[i] = Lmin
                mp[i] = p + ap[Larg]
    return dist, mp, nn


def sphere_uvd(X, norm=False):
    uvd = X.copy()
    uvd[:,0] = np.arccos(X[:,0] / np.linalg.norm(X[:,(0,1)], axis=1)) \
        * (1*(X[:,1] >= 0) - (X[:,1] < 0))
    uvd[:,1] = np.linalg.norm(X[:,:2], axis=1)
    uvd[:,2] = np.arcsin(uvd[:,2] / uvd[:,1])
    
    print(np.any(np.isnan(uvd)))
    
    if norm is False:
        pass
    elif norm is True:
        uvd -= uvd.min(axis=0)
        uvd /= uvd.max(axis=0)
    else:
        uvd[:,norm] -= uvd[:,norm].min(axis=0)
        uvd[:,norm] /= uvd[:,norm].max(axis=0)
    return uvd


def quantirize(P, m=1):
    k = P[0]
    p0 = P[1]
    Q = [k]
    p0k, mag = norm(p0 - k, True)
    
    for p1 in P[2:]:
        pp, ppm = norm(p1 - p0, True)
        mag += ppm
        
        p1k = norm(p1 - k)
        dot = np.dot(p0k, p1k)
        
        if dot < 1 - np.exp(-mag/m):
            #new keypoint detected
            k = p0
            p0 = p1
            p0k = pp
            mag = ppm
            Q.append(k)
        else:
            #update
            p0 = p1
            p0k = p1k
    return np.array(Q)


def mask_planar(eN, fN, Ti_flat, min_dot=0.9, mask=None):
    fN = fN.repeat(3, axis=0)
    if mask is None:
        mask = np.ones(Ti_flat.max()+1, dtype=bool)
    for fn, i in zip(fN, Ti_flat):
        if mask[i]:
            mask[i] &= np.dot(eN[i], fn) <= min_dot
        else:
            pass
    return mask


###TEST nn_point2line
if __name__ == '__main__':
    from argparse import ArgumentParser
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from utils import *
    
    np.random.seed(5)
    X = np.random.randn(10000,3)
    P = np.random.randn(10000,3)
    Xi = np.arange(X.shape[0]).reshape(-1,2)
    X[Xi[:,1]] *= 0.5
    X[Xi[:,1]] += X[Xi[:,0]]
    
    print("Brute force test of nn_point2line")
    delta = time_delta(time())
    dist, mp, nn = nn_point2line(X, Xi, P)
    print("Time", next(delta)) 
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
