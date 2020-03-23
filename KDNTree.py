
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
            self.data = (Xi, x, spatial.magnitude(x))
            self.leaf_size = Xi.shape[0]
        
        def __len__(self):
            return self.leaf_size
        
        def __str__(self):
            return str(len(self))
        
        def query(self, tree, jid, Pi):
            def query_point(X, Xi, xp, Pi):
                L = spatial.magnitude(xp).min(axis=-1)
                Lmin = L < tree.L[jid, Pi]
                if np.any(Lmin):
                    Pi = Pi[Lmin]
                    tree.L[jid,Pi] = L[Lmin]
                    tree.nn[jid, Pi, 0] = Xi[Lmin]
                    tree.nn[jid, Pi, 1:] = -1 
                    tree.mp[jid, Pi] = X[Lmin]
                    tree.done[Xi[Lmin]] = True
            
            def query_line(PX, x, Xi, a, Pi):
                mp = PX + x * a
                L = spatial.magnitude(mp)
                L = L.min(axis=-1)
                Lmin = L < tree.L[jid, Pi]
                if np.any(Lmin):
                    Pi = Pi[Lmin]
                    tree.L[jid, Pi] = L[Lmin]
                    tree.nn[jid, Pi] = Xi
                    tree.mp[jid, Pi] = tree.P[Pi] + mp[Lmin]
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
        def __init__(self, Xi, norm, depth):
            self.Xi = Xi
            self.norm = norm
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
            a = np.sum(np.dot(X - self.mean, self.norm) > 0.0, axis=-1)
            
            left = self.Xi[a==0]
            center = self.Xi[a==1]
            right = self.Xi[a==2]
            
            if self.left:
                pass
            elif left.shape[0] > tree.leaf_size:
                self.left = KDNTree.Node(left, np.roll(self.norm, 1), self.depth+1)
            elif len(left):
                self.left = KDNTree.Leaf(tree, left)
            
            if self.center:
                pass
            elif center.shape[0] > tree.leaf_size and center.shape[0] != self.Xi.shape[0]:
                self.center = KDNTree.Node(center, np.roll(self.norm, 1), self.depth+1)
            elif len(center):
                self.center = KDNTree.Leaf(tree, center)
            
            if self.right:
                pass
            elif right.shape[0] > tree.leaf_size:
                self.right = KDNTree.Node(right, np.roll(self.norm, 1), self.depth+1)
            elif len(right):
                self.right = KDNTree.Leaf(tree, right)
            
        
        def query(self, tree, jid, Pi):
            self.__expand__(tree)
            a = np.dot(tree.P[Pi] - self.mean, self.norm)
            both = a**2 < tree.L[jid, Pi]
            left = a < 0
            right = ~left | both
            left |= both
            
            if self.center:
                yield self.center.query(tree, jid, Pi)
            
            if self.left and np.any(left):
                yield self.left.query(tree, jid, Pi[left])
            
            if self.right and np.any(right):
                yield self.right.query(tree, jid, Pi[right])


    def __init__(self, X, Xi, j=1, leaf_size=None):
        self.X = X
        self.leaf_size = leaf_size if leaf_size else 1 + X.shape[0] / 100
        norm = np.eye(X.shape[1])[0]
        self.roots = [KDNTree.Node(xi, norm, 0) for xi in np.array_split(Xi, j)]
        self.N = Xi.shape[-1]
        self.done = np.zeros(X.shape[0], dtype=bool)
    
    def __str__(self):
        return "**KDNtree**\n  Leaf Size: {}\n".format(self.leaf_size) + \
            "\n".join(["  Root:{}".format(str(root)) for root in self.roots])
    
    def query(self, P, batch_size=None, callback=None):
        j = len(self.roots)
        self.run = True
        self.P = P
        self.mp = np.zeros((j, *P.shape))
        self.nn = -np.ones((j, P.shape[0], self.N), dtype=int)
        self.L = np.zeros((j, P.shape[0])) + np.inf
        Pi = np.arange(P.shape[0])
        
        def job(jid):
            _Pi = np.split(Pi, Pi.shape[0]/batch_size if batch_size else 1)
            
            for pi in _Pi:
                stack = deque([None])
                stack.append(self.roots[jid].query(self, jid, pi))
                node = stack.pop()
                while node and self.run:
                    for n in node:
                        if n:
                            stack.append(n)
                    if callback:
                        callback(self)
                    node = stack.pop()
        
        jobs = [Thread(target=job, args=[jid]) for jid, _ in enumerate(self.roots)]
        for j in jobs:
            j.start()
        
        try:
            for j in jobs:
                while j.is_alive():
                    j.join(0.1)
        except KeyboardInterrupt as e:
            self.run = False
            raise e
        
        Larg = self.L.argmin(axis=0)
        self.L = self.L[Larg, Pi]
        self.mp = self.mp[Larg, Pi]
        self.nn = self.nn[Larg, Pi]
        
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
    tree = KDNTree(X, Xi, j=args.jobs, leaf_size=args.leaf_size)
    
    print("\n0%                      |50%                     |100%")
    dist, mp, nn = tree.query(P, batch_size=args.batch_size, callback=callback)
    
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