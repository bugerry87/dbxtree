import numpy as np
from collections import deque
import spatial


class KDNTree:
    class Leaf:
        def __init__(self, tree, Xi):
            X = tree.X[Xi]
            x = X[:,range(1,X.shape[1])] - X[:,np.zeros(X.shape[1]-1, dtype=int)]
            self.data = zip(X, Xi, x, spatial.magnitude(x))
            self.leaf_size = Xi.shape[0]
        
        def __len__(self):
            return self.leaf_size
        
        def __str__(self):
            return str(len(self))
        
        def query(self, tree, P, mask):
            def query_point(X, Xi, xp, sub_mask):
                L_mask = mask.copy()
                L_mask[mask] = sub_mask
                L = spatial.magnitude(xp).min(axis=-1)
                Lmin = L < tree.L[L_mask]
                if np.any(Lmin):
                    L_mask[L_mask] = Lmin
                    tree.L[L_mask] = L[Lmin]
                    tree.nn[L_mask, 0] = Xi[Lmin]
                    tree.nn[L_mask, 1:] = -1 
                    tree.mp[L_mask] = X[Lmin]
            
            def query_line(PX, x, Xi, a, sub_mask):
                L_mask = mask.copy()
                L_mask[mask] = sub_mask
                mp = PX + x * a
                L = spatial.magnitude(mp)
                L = L.min(axis=-1)
                Lmin = L < tree.L[L_mask]
                if np.any(Lmin):
                    L_mask[L_mask] = Lmin
                    tree.L[L_mask] = L[Lmin]
                    tree.nn[L_mask] = Xi
                    tree.mp[L_mask] = P[sub_mask][Lmin] + mp[Lmin]
        
            for X, Xi, x, m in self.data:
                XP = P - X[0]
                a = (np.sum(XP * x, axis=-1) / m).T
                
                point = (a <= 0).astype(int) + (a >= 1)*2
                line = point == 0
                face = line.prod(axis=-1).astype(bool)
                
                if np.any(point):
                    i = point[point != 0] - 1
                    point = np.any(point, axis=-1)
                    query_point(X[i], Xi[i], XP[point], point)
                
                if np.any(line):
                    i = np.where(line)[1]
                    line = np.any(line, axis=-1)
                    query_line(-XP[line], x[i], Xi, a[line], line)
                    
                if np.any(face):
                    pass

    class Node:
        def __init__(self, Xi, norm, depth):
            self.Xi = Xi
            self.norm = norm
            self.depth = depth
            self.left = None
            self.center = None
            self.right = None
        
        def __len__(self):
            return self.depth
        
        def __str__(self):
            return "{:-<4}-+---Left:{}\
                \n          {}  \\_Center:{}\
                \n          {}  \\__Right:{}".format(
                self.depth,
                self.left,
                "  |           " * self.depth,
                self.center,
                "  |           " * self.depth,
                self.right)
        
        def __unfold__(self, tree):
            X = tree.X[self.Xi]
            self.mean = X.reshape(-1, 3).mean(axis=0)
            a = np.sum(np.dot(X - self.mean, self.norm) >= 0.0, axis=-1)
            
            left = self.Xi[a==0]
            center = self.Xi[a==1]
            right = self.Xi[a==2]
            
            if self.left:
                pass
            elif left.shape[0] > tree.leaf_size:
                self.left = KDNTree.Node(left, np.roll(self.norm, 1), self.depth+1)
            elif len(left):
                self.left = KDNTree.Leaf(tree, left)
            else:
                self.left = None
            
            if self.center:
                pass
            elif center.shape[0] > tree.leaf_size:
                self.center = KDNTree.Node(center, np.roll(self.norm, 1), self.depth+1)
            elif len(center):
                self.center = KDNTree.Leaf(tree, center)
            else:
                self.center = None
            
            if self.right:
                pass
            elif right.shape[0] > tree.leaf_size:
                self.right = KDNTree.Node(right, np.roll(self.norm, 1), self.depth+1)
            elif len(right):
                self.right = KDNTree.Leaf(tree, right)
            else:
                self.right = None
        
        def query(self, tree, P, mask):
            self.__unfold__(tree)
            a = np.dot(P - self.mean, self.norm)
            both = a**2 > tree.L[mask]
            left = a < 0
            right = ~left | both
            left |= both
            
            if self.center:
                yield self.center.query(tree, P, mask)
            
            if self.left and np.any(left):
                L_mask = mask.copy()
                L_mask[mask] = left
                yield self.left.query(tree, P[left], L_mask)
            
            if self.right and np.any(right):
                L_mask = mask.copy()
                L_mask[mask] = right
                yield self.right.query(tree, P[right], L_mask)


    def __init__(self, X, Xi, leaf_size=None):
        self.X = X
        self.Xi = Xi
        self.leaf_size = leaf_size if leaf_size else 1 + X.shape[0] / 100
        norm = np.eye(X.shape[1])[0]
        self.root = KDNTree.Node(Xi, norm, 0)
        self.N = Xi.shape[-1]
    
    def __str__(self):
        return "**KDNtree**\n  Leaf Size: {}\n  Root:{}".format(self.leaf_size, str(self.root))
    
    def query(self, P):
        self.mp = np.zeros(P.shape)
        self.nn = -np.ones((P.shape[0], self.N), dtype=int)
        self.L = np.zeros(P.shape[0]) + np.inf
        mask = np.ones(P.shape[0], dtype=bool)
        
        stack = deque([None])
        stack.append(self.root.query(self, P, mask))
        node = stack.pop()
        while node:
            for n in node:
                if n:
                    stack.append(n)
            node = stack.pop()
            
        return self.L, self.mp, self.nn


###TEST
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from utils import *
    
    np.random.seed(5)
    X = np.random.randn(10000,3)
    P = np.random.randn(10000,3)
    Xi = np.arange(X.shape[0]).reshape(-1,2)
    X[Xi[:,1]] *= 0.5
    X[Xi[:,1]] += X[Xi[:,0]]
    
    print("KDNTree")
    delta = time_delta(time())
    tree = KDNTree(X, Xi, leaf_size=100)
    dist, mp, nn = tree.query(P)
    print("Query time:", next(delta))
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
    print(tree)