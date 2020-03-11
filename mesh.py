'''
Mesh operation functions for this project.

Author: Gerald Baulig
'''

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial import Delaunay


def norm(X, magnitude=False):
    if len(X.shape) == 2:
        m = np.linalg.norm(X, axis=1)[:,None]
    else:
        m = np.linalg.norm(X)
    n = X / m
    if magnitude:
        return n, m
    else:
        return n


class PlaneTree:
    class Node:
        def __init__(self, a, b, i):
            self.a, self.b, self.n, self.i, = (a, b, b-a, i)
            self.m = np.linalg.norm(self.n)
            self.left, self.center, self.right = (None, None, None)
        
        def add_node(self, a, b, i):
            aa = np.sum(self.n * a - self.a)
            ab = np.sum(self.n * b - self.a)
            ba = np.sum(-self.n * a - self.b)
            bb = np.sum(-self.n * b - self.b)
            
            if aa <= 0.0 or ab <= 0.0:
                if self.left:
                    self.left.add_node(a, b, i)
                else:
                    self.left = PlaneTree.Node(a, b, i)
              
            if ba <= 0.0 or bb <= 0.0:
                if self.right:
                    self.right.add_node(a, b, i)
                else:
                    self.right = PlaneTree.Node(a, b, i)
                
            if aa * ab > 0.0 and ba * bb > 0.0:
                if self.center:
                    self.center.add_node(a, b, i)
                else:
                    self.center = PlaneTree.Node(a, b, i)
        
        def query(self, P, tree, mask):
            aP = P - self.a
            bP = P - self.b
            a = np.sum(self.n * aP, axis=1)
            b = np.sum(self.n * bP, axis=1)
            left = mask & False
            left[mask] = a <= 0.0
            lm = left[mask]
            if any(left):
                m = np.linalg.norm(aP[lm], axis=1)
                Lmin = np.min((tree.L[left], m), axis=0).astype(bool)
                left_Lmin = left & False
                left_Lmin[left] = Lmin
                if any(Lmin):
                    tree.L[left_Lmin] = m[Lmin]
                    tree.mp[left_Lmin] = self.a
                    tree.nn[left_Lmin] = (self.i[0], self.i[0])
                if self.left:
                    self.left.query(P[lm], tree, left)
            
            right = mask & False
            right[mask] = b >= 0.0
            rm = right[mask]
            if any(right):
                m = np.linalg.norm(bP[rm], axis=1)
                Lmin = np.min((tree.L[right], m), axis=0).astype(bool)
                right_Lmin = right & False
                right_Lmin[right] = Lmin
                if any(Lmin):
                    tree.L[right_Lmin] = m[Lmin]
                    tree.mp[right_Lmin] = self.b
                    tree.nn[right_Lmin] = (self.i[1], self.i[1])
                if self.right:
                    self.right.query(P[rm], tree, right)
            
            center = mask & False
            center[mask] = ~lm & ~rm
            cm = center[mask]
            if any(center):
                n, m = norm((self.n/self.m)[None,:] * a[cm][:,None] + self.a - P[cm], True)
                Lmin = np.min((tree.L[center], m[:,0]), axis=0).astype(bool)
                if any(Lmin):
                    center_Lmin = center & False
                    center_Lmin[center] = Lmin
                    tree.L[center_Lmin] = m[Lmin,0]
                    tree.mp[center_Lmin] = P[cm][Lmin] + n[Lmin] * m[Lmin]
                    tree.nn[center_Lmin] = self.i
                if self.right:
                    self.right.query(P[cm], tree, center)

    def __init__(self, X, Xi):
        self.root = None
        for ai, bi in zip(Xi[:,0], Xi[:,1]):
            a, b = (X[ai], X[bi])
            self.add_node(a, b, (ai, bi))
    
    def add_node(self, a, b, i):
        if self.root:
            self.root.add_node(a, b, i)
        else:
            self.root = PlaneTree.Node(a, b, i)
    
    def query(self, P):
        self.mp = np.zeros(P.shape)
        self.nn = -np.ones((P.shape[0],2), dtype=int)
        self.L = np.zeros(P.shape[0]) + np.inf
        mask = np.ones(P.shape[0], dtype=bool)
        self.root.query(P, self, mask)
        return self.L, self.mp, self.nn


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


def nn_point2point(X, Y):
    return KDTree(Y).query(X)


def nn_point2line(X, Xi, P):
    total = time_delta(time())
    delta = time_delta(time())
    nn = -np.ones((P.shape[0],2), dtype=int)
    dist, nn[:,0] = KDTree(X).query(P)
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
    
    np.random.seed(5)
    X = np.random.randn(5,3)
    P = np.random.randn(4,3)
    Xi = np.array((range(X.shape[0]-1), range(1,X.shape[0]))).T
    
    #print("Brute force")
    #dist, mp, nn = nn_point2line(X, Xi, P)
    #print("Mean loss:", dist.mean())
    
    delta = time_delta(time())
    tree = PlaneTree(X, Xi)
    print("Tree setup:", next(delta))
    dist, mp, nn = tree.query(P)
    print("Inference time:", next(delta))
    print("Mean loss:", dist.mean())
    
    fig = plt.figure()
    ax = fig.add_subplot((111), projection='3d')
    seg = np.hstack((X[Xi[:,0]], X[Xi[:,1]]-X[Xi[:,0]]))
    x, y, z, u, v, w = zip(*seg)
    mp -= P
    
    ax.quiver(x, y, z, u, v, w)
    ax.scatter(P[:,0],P[:,1],P[:,2], color='r')
    ax.quiver(P[:,0],P[:,1],P[:,2],mp[:,0],mp[:,1],mp[:,2], color='g')
    
    ax.set_xlim3d(-3, 3)
    ax.set_ylim3d(-3, 3)
    ax.set_zlim3d(-3, 3)
    plt.show()