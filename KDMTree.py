
from mesh import *


class KDMTree:
    class Node:
        def __init__(self, pos, norm):
            self.pos = pos
            self.norm = norm
            self.left = None
            self.right = None
        
        def set_payload(self, tree, Xi):
            if Xi.shape[0] > tree.leaf_size:
                a = np.sum(np.dot(tree.X[Xi], self.norm) >= 0.0, axis=0)
                left = Xi[a==0]
                self.payload = Xi[a==1]
                right = Xi[a==2]
                
                if left.shape[0]:
                    if not self.left:
                        x = tree.X[left].reshape(-1, 3)
                        pos = x.mean(axis=0)
                        norm = np.roll(self.norm, 1)
                        self.left = KDMTree(pos, norm)
                    self.left.set_payload(left)
                
                if right.shape[0]:
                    if not self.right:
                        x = tree.X[right].reshape(-1, 3)
                        pos = x.mean(axis=0)
                        norm = np.roll(self.norm, 1)
                        self.right = KDMTree(pos, norm)
                    self.right.set_payload(right)
            else:
                self.payload = Xi
                


    def __init__(self, X, Xi):
        self.root = None
        for ai, bi in zip(Xi[:,0], Xi[:,1]):
            a, b = (X[ai], X[bi])
            node = PlaneTree.Node(a, b, (ai, bi))
            self.add_node(node)
    
    def add_node(self, node):
        if self.root:
            self.root.add_node(node)
        else:
            self.root = node
    
    def query(self, P):
        self.mp = np.zeros(P.shape)
        self.nn = -np.ones((P.shape[0],2), dtype=int)
        self.L = np.zeros(P.shape[0]) + np.inf
        mask = np.ones(P.shape[0], dtype=bool)
        self.root.query(P, self, mask)
        return self.L, self.mp, self.nn


###TEST
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from utils import *
    
    np.random.seed(5)
    X = np.random.randn(10000,3)
    P = np.random.randn(10000,3)
    Xi = np.array((range(X.shape[0]-1), range(1,X.shape[0]))).T
    AB = np.sum((X[Xi[:,1]] - X[Xi[:,0]])**2, axis=-1)
    AB = np.argsort(-AB)
    Xi = Xi[AB]
    
    print("Brute force")
    dist, mp, nn = nn_point2line(X, Xi, P)
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