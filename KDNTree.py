
#BuildIn
from threading import Thread, Lock
from collections import deque
import time

#Installed
import numpy as np

#Local
import spatial


class KDNTree:
    class Leaf:
        def __init__(self, tree, Xi):
            X = tree.X[Xi]
            x = X[:,range(1,X.shape[1])] - X[:,np.zeros(X.shape[1]-1, dtype=int)]
            self.data = (X, Xi, x, spatial.magnitude(x))
            self.leaf_size = Xi.shape[0]
        
        def __len__(self):
            return self.leaf_size
        
        def __str__(self):
            return str(len(self))
        
        def query(self, tree, Pi):
            def query_point(X, Xi, xp, Pi):
                L = spatial.magnitude(xp).min(axis=-1)
                Lmin = L < tree.L[Pi]
                if np.any(Lmin):
                    Pi = Pi[Lmin]
                    tree.L[Pi] = L[Lmin]
                    tree.nn[Pi, 0] = Xi[Lmin]
                    tree.nn[Pi, 1:] = -1 
                    tree.mp[Pi] = X[Lmin]
                    tree.done[Xi[Lmin]] = True
            
            def query_line(PX, x, Xi, a, Pi):
                mp = PX + x * a
                L = spatial.magnitude(mp)
                L = L.min(axis=-1)
                Lmin = L < tree.L[Pi]
                if np.any(Lmin):
                    Pi = Pi[Lmin]
                    tree.L[Pi] = L[Lmin]
                    tree.nn[Pi] = Xi
                    tree.mp[Pi] = tree.P[Pi] + mp[Lmin]
                    tree.done[Xi] = True
        
            for X, Xi, x, m in zip(*self.data):
                XP = tree.P[Pi] - X[0]
                a = (np.sum(XP * x, axis=-1) / m).T
                
                point = np.nonzero((a <= 0, a >= 1))
                line = np.nonzero((a > 0) * (a < 1))
                #face = line.prod(axis=-1).astype(bool)
                
                if len(point[0]):
                    i = point[0] * (1+point[2])
                    point = Pi[point[1]]
                    xp = tree.P[point] - X[i]
                    query_point(X[i], Xi[i], xp, point)
                
                if len(line[0]):
                    i = line[1]
                    pi = Pi[line[0]]
                    query_line(-XP[line[0]], x[i], Xi, a[line[0]], pi)
                        
                #if len(face):
                #    pass

    class Node:
        def __init__(self, Xi, norm, depth):
            self.Xi = Xi
            self.norm = norm
            self.depth = depth
            self.left = None
            self.center = None
            self.right = None
            self.__expanded = False
            self.__lock_expand = Lock()
        
        def __len__(self):
            return self.depth
        
        def __str__(self):
            return "{:-<4}-+---Left:{}\
                \n          {}  \\_Center:{}\
                \n          {}  \\__Right:{}".format(
                '-',
                self.left,
                "  |           " * self.depth,
                self.center,
                "  |           " * self.depth,
                self.right)
        
        def __expand__(self, tree):
            if self.__expanded:
                return
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
            self.__expanded = True
        
        def query(self, tree, Pi):
            if not self.__expanded:
                self.__lock_expand.acquire()
                self.__expand__(tree)
                self.__lock_expand.release()
            
            a = np.dot(tree.P[Pi] - self.mean, self.norm)
            both = a**2 < tree.L[Pi]
            left = a < 0
            right = ~left | both
            left |= both
            
            if self.center:
                yield self.center.query(tree, Pi)
            
            if self.left and np.any(left):
                yield self.left.query(tree, Pi[left])
            
            if self.right and np.any(right):
                yield self.right.query(tree, Pi[right])


    def __init__(self, X, Xi, leaf_size=None):
        self.X = X
        self.Xi = Xi
        self.leaf_size = leaf_size if leaf_size else 1 + X.shape[0] / 100
        norm = np.eye(X.shape[1])[0]
        self.root = KDNTree.Node(Xi, norm, 0)
        self.N = Xi.shape[-1]
        self.done = np.zeros(X.shape[0], dtype=bool)
    
    def __str__(self):
        return "**KDNtree**\n  Leaf Size: {}\n  Root:{}".format(self.leaf_size, str(self.root))
    
    def query(self, P, j=1, batch_size=None, callback=None):
        self.P = P
        self.mp = np.zeros(P.shape)
        self.nn = -np.ones((P.shape[0], self.N), dtype=int)
        self.L = np.zeros(P.shape[0]) + np.inf
        self.run = True
        Pi = np.arange(P.shape[0])
        Pi = np.split(Pi, j)
        
        def job(Pi):
            Pi = np.split(Pi, Pi.shape[0]/batch_size if batch_size else 1)
            for pi in Pi:
                stack = deque([None])
                stack.append(self.root.query(self, pi))
                node = stack.pop()
                while node and self.run:
                    for n in node:
                        if n:
                            stack.append(n)
                    if callback:
                        callback(self)
                    node = stack.pop()
        
        jobs = [Thread(target=job, args=[pi]) for pi in Pi]
        for j in jobs:
            j.start()
        
        try:
            for j in jobs:
                while j.is_alive():
                    j.join(0.1)
        except KeyboardInterrupt as e:
            self.run = False
            raise e
        
        return self.L, self.mp, self.nn

###TEST
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from utils import *
    
    print_lock = Lock()
    last = 0
    def callback(tree):
        global last
        print_lock.acquire()
        curr = int(tree.done.mean() * 50)
        dif = curr - last
        print(u"\u2588" * dif, end='', flush=dif)
        last = curr
        print_lock.release()
    
    np.random.seed(20)
    X = np.random.randn(10000,3)
    P = np.random.randn(100000,3)
    Xi = np.arange(X.shape[0]).reshape(-1,2)
    X[Xi[:,1]] *= 0.5
    X[Xi[:,1]] += X[Xi[:,0]]
    
    print("KDNTree")
    print("Model size:", X.shape)
    print("Query size:", P.shape)
    
    delta = time_delta(time())
    tree = KDNTree(X, Xi, leaf_size=10)
    
    print("\n0%                      |50%                     |100%")
    dist, mp, nn = tree.query(P, j=1, batch_size=None, callback=callback)
    
    print("\nQuery time:", next(delta))
    print("Mean loss:", dist.mean())
    
    fig = plt.figure()
    ax = fig.add_subplot((111), projection='3d')
    seg = np.hstack((X[Xi[:,0]], X[Xi[:,1]]-X[Xi[:,0]]))
    x, y, z, u, v, w = zip(*seg)
    mp -= P
    point = nn[:,1] == -1
    
    ax.quiver(x, y, z, u, v, w)
    ax.scatter(P[:,0],P[:,1],P[:,2], color='r')
    ax.quiver(P[point,0],P[point,1],P[point,2],mp[point,0],mp[point,1],mp[point,2], color='g')
    ax.quiver(P[~point,0],P[~point,1],P[~point,2],mp[~point,0],mp[~point,1],mp[~point,2], color='y')
    
    ax.set_xlim3d(-3, 3)
    ax.set_ylim3d(-3, 3)
    ax.set_zlim3d(-3, 3)
    plt.show()
    print(tree)