import numpy as np
import spatial


class KDNTree:
    class Node:
        def __init__(self, tree, Xi, norm):
            self.norm = norm
            self.left = None
            self.right = None
            self.center = []
        
            X = tree.X[Xi]
            self.mean = X.reshape(-1, 3).mean(axis=0)
            a = np.sum(np.dot(X - self.mean, self.norm) >= 0.0, axis=-1)
            
            if Xi.shape[0] > tree.leaf_size:
                left = Xi[a==0]
                right = Xi[a==2]
                X = X[a==1]
                Xi = Xi[a==1]
                
                if len(left):
                    self.left = KDNTree.Node(tree, left, np.roll(self.norm, 1))
                
                if len(right):
                    self.right = KDNTree.Node(tree, right, np.roll(self.norm, 1))
                    
            if len(X):
                x = X[:,np.zeros(X.shape[1]-1, dtype=int)] - X[:,range(1,X.shape[1])]
                self.center = zip(X, Xi, x, spatial.magnitude(x))
        
        def query(self, tree, P, mask):
            def query_point(X, Xi, xp, sub_mask):
                L_mask = mask.copy()
                L_mask[mask] = sub_mask
                L = spatial.magnitude(xp).min(axis=-1)
                Lmin = L < tree.L[L_mask]
                if np.any(Lmin):
                    L_mask[L_mask] = Lmin
                    tree.L[L_mask] = L[Lmin]
                    tree.nn[L_mask] = Xi
                    tree.mp[L_mask] = X[Lmin]
            
            def query_line(PX, x, Xi, a, sub_mask):
                L_mask = mask.copy()
                L_mask[mask] = sub_mask
                mp = PX + x * np.abs(a)
                L = spatial.magnitude(mp)
                Larg = L.argmin(axis=-1)
                L = L.min(axis=-1)
                Lmin = L < tree.L[L_mask]
                if np.any(Lmin):
                    L_mask[L_mask] = Lmin
                    tree.L[L_mask] = L[Lmin]
                    tree.nn[L_mask] = Xi
                    tree.mp[L_mask] = P[sub_mask][Lmin] + mp[Larg][Lmin]
        
            for X, Xi, x, m in self.center:
                XP = P - X[0]
                a = np.sum(XP * x, axis=-1) / m
                a = a.T
                
                point = 1 - (a > 0) + (a >= 1)
                line = point == 0
                face = line.prod(axis=-1).astype(bool)
                
                if np.any(point):
                    i = point[point > 0] - 1
                    point = np.any(point, axis=-1)
                    query_point(X[i], Xi, XP[point], point)
                
                if np.any(line):
                    i = np.where(line)[1]
                    line = np.any(line, axis=-1)
                    query_line(-XP[line], x[i], Xi, a[line], line)
                    
                if np.any(face):
                    pass
                
            a = np.dot(P - self.mean, self.norm)
            both = np.abs(a) > tree.L[mask]
            left = a < 0
            right = ~left | both
            left |= both
            
            if np.any(left) and self.left:
                L_mask = mask.copy()
                L_mask[mask] = left
                self.left.query(tree, P[left], L_mask)
            
            if np.any(right) and self.right:
                L_mask = mask.copy()
                L_mask[mask] = right
                self.right.query(tree, P[right], L_mask)


    def __init__(self, X, Xi, leaf_size=10):
        self.X = X
        self.Xi = Xi
        self.leaf_size = leaf_size
        norm = np.eye(X.shape[1])[0]
        self.root = KDNTree.Node(self, Xi, norm)
    
    def query(self, P):
        self.mp = np.zeros(P.shape)
        self.nn = -np.ones((P.shape[0], self.Xi.shape[-1]), dtype=int)
        self.L = np.zeros(P.shape[0]) + np.inf
        mask = np.ones(P.shape[0], dtype=bool)
        self.root.query(self, P, mask)
        return self.L, self.mp, self.nn


###TEST
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from utils import *
    
    np.random.seed(10)
    X = np.random.randn(4,3)
    P = np.random.randn(8,3)
    Xi = np.array((range(X.shape[0]-1), range(1,X.shape[0]))).T
    AB = np.sum((X[Xi[:,1]] - X[Xi[:,0]])**2, axis=-1)
    AB = np.argsort(-AB)
    Xi = Xi[AB]
    
    print("KDNTree")
    delta = time_delta(time())
    dist, mp, nn = KDNTree(X, Xi, leaf_size=1).query(P)
    print("Time", next(delta)) 
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