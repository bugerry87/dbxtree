
#BuildIn
from multiprocessing import Pool, Lock
from collections import deque
import time
import signal

#Installed
import numpy as np

#Local
import spatial


def __job__(data):
    tree, root = data
    tree.root = root
    _Pi = np.split(tree.Pi, tree.Pi.shape[0]/tree.batch_size if tree.batch_size else 1)
    
    for pi in _Pi:
        stack = deque([None])
        stack.append(root.query(tree, pi))
        node = stack.pop()
        while node:
            for n in node:
                if n:
                    stack.append(n)
            #if self.callback:
            #    self.callback(self)
            node = stack.pop()
    return tree


class KDNTree:
    class Leaf:
        def __init__(self, tree, Xi):
            X = tree.X[Xi]
            x = X[:,range(1,X.shape[1])] - X[:,np.zeros(X.shape[1]-1, dtype=int)]
            self.data = (Xi, x, spatial.magnitude(x))
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
        
            for Xi, x, m in zip(*self.data):
                X = tree.X[Xi]
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
        def __init__(self, Xi, depth):
            self.Xi = Xi
            self.depth = depth
            self.left = None
            self.center = None
            self.right = None
            self.__expanded = False
        
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
            self.__expanded = True
            
            X = tree.X[self.Xi]
            self.mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
            X = X - self.mean
            
            self.mag = 0.0
            while self.mag == 0.0:
                i = np.random.choice(X.shape[0], 2)
                c = np.cross(X[i[0],0], X[i[1],1])
                self.mag = spatial.magnitude(c)
            self.norm = c
            a = np.sum(np.dot(X, self.norm) > 0.0, axis=-1)
            
            left = self.Xi[a==0]
            center = self.Xi[a==1]
            right = self.Xi[a==2]
            
            if self.left:
                pass
            elif left.shape[0] > tree.leaf_size:
                self.left = KDNTree.Node(left, self.depth+1)
            elif len(left):
                self.left = KDNTree.Leaf(tree, left)
            
            if self.center:
                pass
            elif center.shape[0] > tree.leaf_size:
                self.center = KDNTree.Node(center, self.depth+1)
            elif len(center):
                self.center = KDNTree.Leaf(tree, center)
            
            if self.right:
                pass
            elif right.shape[0] > tree.leaf_size:
                self.right = KDNTree.Node(right, self.depth+1)
            elif len(right):
                self.right = KDNTree.Leaf(tree, right)
        
        def query(self, tree, Pi):
            self.__expand__(tree)
            a = np.dot(tree.P[Pi] - self.mean, self.norm) / self.mag
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


    def __init__(self, X, Xi, j=1, leaf_size=None, batch_size=None, callback=None):
        self.X = X
        self.roots = [KDNTree.Node(xi, 0) for xi in np.array_split(Xi, j)]
        self.batch_size = batch_size
        #self.callback = callback
        self.leaf_size = leaf_size if leaf_size else 1 + X.shape[0] / 100
        self.N = Xi.shape[-1]
        self.done = np.zeros(X.shape[0], dtype=bool)
    
    def __str__(self):
        return "**KDNtree**\n  Leaf Size: {}\n".format(self.leaf_size) + \
            "\n".join(["  Root:{}".format(str(root)) for root in self.roots])
    
    def query(self, P):
        j = len(self.roots)
        self.P = P
        self.mp = np.zeros(P.shape)
        self.nn = -np.ones((P.shape[0], self.N), dtype=int)
        self.L = np.zeros(P.shape[0]) + np.inf
        self.Pi = np.arange(P.shape[0])
        
        pool = Pool(j)
        trees = pool.map(__job__, zip([self]*j, self.roots))
        
        self.roots = [t.root for t in trees]
        L = np.array([t.L for t in trees])
        mp = np.array([t.mp for t in trees])
        nn = np.array([t.nn for t in trees])
        
        Larg = L.argmin(axis=0)
        self.L = L[Larg, self.Pi]
        self.mp = mp[Larg, self.Pi]
        self.nn = nn[Larg, self.Pi]
        
        return self.L, self.mp, self.nn

###TEST
if __name__ == '__main__':
    from argparse import ArgumentParser
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from utils import *
    
    def init_argparse(parents=[]):
        ''' init_argparse(parents=[]) -> parser
        Initialize an ArgumentParser for this module.
        
        Args:
            parents: A list of ArgumentParsers of other scripts, if there are any.
            
        Returns:
            parser: The ArgumentParsers.
        '''
        parser = ArgumentParser(
            #description="Demo for embedding data via LDA",
            parents=parents
            )
        
        parser.add_argument(
            '--model_size', '-m',
            metavar='INT',
            type=int,
            default=10000
            )
        
        parser.add_argument(
            '--query_size', '-q',
            metavar='INT',
            type=int,
            default=10000
            )
        
        parser.add_argument(
            '--batch_size', '-b',
            metavar='INT',
            type=int,
            default=0
            )
        
        parser.add_argument(
            '--leaf_size', '-l',
            metavar='INT',
            type=int,
            default=100
            )
        
        parser.add_argument(
            '--jobs', '-j',
            metavar='INT',
            type=int,
            default=1
            )
        
        parser.add_argument(
            '--seed', '-s',
            metavar='INT',
            type=int,
            default=0
            )
        
        return parser
    
    args, _ = init_argparse().parse_known_args()
    print_lock = Lock()
    last = 0
    def callback(tree):
        global last
        print_lock.acquire()
        curr = int(tree.done.mean() * 50)
        dif = curr - last
        if curr > last:
            print('#' * dif, end='', flush=True)
        last = curr
        print_lock.release()
    
    np.random.seed(args.seed)
    X = np.random.randn(args.model_size,3)
    P = np.random.randn(args.query_size,3)
    Xi = np.arange(X.shape[0]).reshape(-1,2)
    X[Xi[:,1]] *= 0.5
    X[Xi[:,1]] += X[Xi[:,0]]
    
    print("KDNTree")
    print("Model size:", X.shape)
    print("Query size:", P.shape)
    
    delta = time_delta(time())
    tree = KDNTree(
        X, Xi,
        args.jobs,
        args.leaf_size,
        args.batch_size,
        callback=callback
        )
    
    print("\n0%                      |50%                     |100%")
    dist, mp, nn = tree.query(P)
    
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