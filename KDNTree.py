from mesh import *


class KDNTree:
    class Node:
        def __init__(self, tree, Xi, norm):
            self.norm = norm
            self.left = None
            self.right = None
        
            X = tree.X[Xi]
            self.mean = self.X.reshape(-1, 3).mean(axis=0)
            n = self.X - self.mean
            x = self.X[:,0] - self.X[:,1:]
            m = np.dot(n, self.norm)
            a = np.sum(m >= 0.0, axis=0)
            
            if Xi.shape[0] > tree.leaf_size:
                left = Xi[a==0]
                right = Xi[a==2]
                X = X[a==1]
                Xi = Xi[a==1]
                n = n[a==1]
                x = [a==1]
                
                if left.shape[0]:
                    self.left = Node(tree, left, np.roll(self.norm, 1))
                
                if right.shape[0]:
                    self.right = Node(tree, right, np.roll(self.norm, 1))
            self.center = (X, Xi, x, n, m)
        
        def query(self, tree, P, mask):
            for i, p in enumerate(P):
                AP = P - A
                BP = P - B
                a = np.sum(AP * AB, axis=1)
                b = np.sum(BP * -AB, axis=1)
                
                head = b <= 0
                body = (a * b) > 0
                tail = a <= 0
                
                if any(head):
                    
                
                if any(body):
                    n, L = norm(AB[m] * a[m][:,None] + A[m] - p, True)
                    Larg = np.argmin(L)
                    Lmin = L[Larg]
                    if Lmin < dist[i]:
                        nn[i] = Xi[m][Larg]
                        dist[i] = Lmin
                        mp[i] = p + n[Larg] * Lmin
                
                if any(tail):
                
                


    def __init__(self, X, Xi, leaf_size=10):
        self.X = X
        self.leaf_size = leaf_size
        norm = np.eye(X.shape[1])[0]
        self.root = Node(self, Xi, norm)
        del self.X
    
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