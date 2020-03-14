
from mesh import *


class PlaneTree:
    class Node:
        def __init__(self, a, b, i):
            self.a, self.b, self.n, self.i, = (a, b, b-a, i)
            self.m = magnitude(self.n)
            self.m_sqrt = np.sqrt(self.m)
            self.n /= self.m_sqrt if self.m != 0.0 else 1.0
            self.left, self.center, self.right = (None, None, None)
            self.depth = 0
        
        def add_node(self, node):
            if node is self:
                return
            else:
                node.depth += 1
        
            aa = np.sum(self.n * node.a - self.a)
            ab = np.sum(self.n * node.b - self.a)
            ba = np.sum(-self.n * node.a - self.b)
            bb = np.sum(-self.n * node.b - self.b)
            assign = True
            
            if aa <= 0.0 or ab <= 0.0:
                if self.left:
                    self.left.add_node(node)
                    assign = False
                else:
                    self.left = node
                    print(node.depth)
              
            if ba <= 0.0 or bb <= 0.0:
                if self.right:
                    if self.left != self.right and assign:
                        self.right.add_node(node)
                        assign = False
                else:
                    self.right = node
                    print(node.depth)
                
            if aa * ab > 0.0 and ba * bb > 0.0:
                if self.center:
                    if self.right != self.center and assign:
                        self.center.add_node(node)
                else:
                    self.center = node
                    print(node.depth)
        
        def query(self, P, tree, mask):
            aP = P - self.a
            bP = P - self.b
            a = np.sum(self.n * aP, axis=1)
            b = np.sum(-self.n * bP, axis=1)
            
            def query_border(a, b, p, x, anode, bnode):
                border = mask & False
                border_Lmin = border.copy()
                border[mask] = a <= 0.0
                bm = border[mask]
                if any(border):
                    m = magnitude(p[bm]).flatten()
                    Lmin = np.argmin((tree.L[border], m), axis=0).astype(bool)
                    if any(Lmin):
                        border_Lmin[:] = False
                        border_Lmin[border] = Lmin
                        tree.L[border_Lmin] = m[Lmin]
                        tree.mp[border_Lmin] = x
                        tree.nn[border_Lmin] = (self.i[0], self.i[0])
                        if self.center:
                            self.center.query(P[bm][Lmin], tree, border_Lmin)
                    if anode:
                        anode.query(P[bm], tree, border)
                    if bnode:
                        m = b[bm]**2
                        Lmin = np.argmin((tree.L[border], m), axis=0).astype(bool)
                        if any(Lmin):
                            border_Lmin[:] = False
                            border_Lmin[border] = Lmin
                            bnode.query(P[bm][Lmin], tree, border_Lmin)
                return bm
            
            lm = query_border(a, b, aP, self.a, self.left, self.right)
            rm = query_border(b, a, bP, self.b, self.right, self.left)
            
            center = mask & False
            center[mask] = ~lm & ~rm
            cm = center[mask]
            if any(center):
                n, m = norm(self.n[None,:] * a[cm][:,None] + self.a - P[cm], True)
                m_2 = m.flatten()**2
                Lmin = np.argmin((tree.L[center], m_2), axis=0).astype(bool)
                if any(Lmin):
                    center_Lmin = center & False
                    center_Lmin[center] = Lmin
                    tree.L[center_Lmin] = m_2[Lmin]
                    tree.mp[center_Lmin] = P[cm][Lmin] + n[Lmin] * m[Lmin]
                    tree.nn[center_Lmin] = self.i
                    if self.left:
                        left = Lmin & (m.flatten() > a[cm])
                        if any(Lmin):
                            center_Lmin[center] = left
                            self.left.query(P[cm][left], tree, center_Lmin)
                    if self.right:
                        right = Lmin & (m.flatten() > b[cm])
                        if any(Lmin):
                            center_Lmin[:] = False
                            center_Lmin[center] = right
                            self.right.query(P[cm][right], tree, center_Lmin)
                if self.center:
                    self.center.query(P[cm], tree, center)

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