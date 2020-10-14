'''
Author: Gerald Baulig
'''

#Global libs
import numpy as np

#Local libs
import kernel


def square_mag(u, v):
    ''' square_mag(a, b) -> squared magnitude
    Calculates the row-wise squared magnitude.
    This function is preferably used for distance comparisons,
    because taking a square root has a high computation time.
        np.sum((a-b)**2, axis=1)
    
    Args:
        u: numpy.ndarray
        v: numpy.ndarray
    Returns:
        The row-wise squared magnitude of a and b.
    '''
    return np.sum((u-v)**2, axis=1)


def kernel_trick(gram, C):
    ''' kernel_trick(gram, C) -> W
    Computes a Weight-Matrix of unsimilarity via "Kernel-Trick".
    
    Args:
        gram: The Gram-Matrix of a Kernel-Function.
        C: An Association-Matrix with cluster assigments per column.
    '''
    N = gram.shape[0]
    K = C.shape[1]
    A = np.sum(C,axis=0)
    ones = np.ones((N, K))
    
    W = np.dot(gram * np.eye(N), ones)
    W -= 2*(np.dot(gram, C) / A)
    W += np.dot(ones, np.dot(np.dot(C.T, gram), C) * np.eye(K))
    W /= (A**2)
    return W


KMEANS_INIT_MODES = ('select', 'mean', 'uniform', 'normal', 'kmeans++', 'limits')
def init_kmeans(X, K, mode='mean'):
    ''' init_kmeans(X, K, mode='free') -> centroids
    Initialize K cluster centroids on dataset X.
    
    Args:
        X: The dataset.
        K: The number of cluster centroids.
        mode: The mode how to initialize the cluster centroids.
            mean = All centroids start (almost) at the mean of the dataset.
                Pro: The result is determenistic.
                Con: Needs more iterations.
            uniform = Uniform random distributed.
                Pro: May work well on uniform distributed datasets.
                Con: Ks may get lost in huge data gaps.
            normal = Normal random distributed.
                Pro: May work well on normal distributed datasets.
                Con: Ks may get lost in huge data gaps.
            select = Selects random points of the dataset.
                Pro: Makes sure that each K has at least one point.
                Con: Not determenistic like all the other random approaches.
            kmeans++ = Selects a random point and rearranges the
                probabillities of selecting the next point according
                to the distance.
                Pro: May have an appropriate distributed of Ks.
                Con: Great effort for a negligible improvment.
    Returns:
        centroids: The cluster centroids.
    '''
    N = X.shape[0]
    d = X.shape[1]
    
    if mode=='mean':
        #One can not take the absolute mean.
        #The first K would claim all points and the algo stuck.
        #Get the centroids and add a small portion of the variance.
        #This ensures a small error and the Ks will swarm out.
        #Pro: The result is determenistic.
        #Con: Needs more iterations.
        centroids = np.tile(np.mean(X, axis=0), (K,1)) + np.var(X, axis=0) * 1e-10
    elif mode=='select':
        centroids = X[np.random.choice(range(N), K),:]
    elif mode=='uniform':
        centroids = np.random.rand(K,d) * np.var(X, axis=0) + np.mean(X, axis=0)
    elif mode=='normal':
        centroids = np.random.randn(K,d) * np.var(X, axis=0) + np.mean(X, axis=0)
    elif mode=='kmeans++':
        #guided by https://stackoverflow.com/questions/5466323/how-exactly-does-k-means-work
        centroids = np.zeros((K,d))
        centroids[0,:] = X[np.random.choice(range(N), 1),:] #pick one
        for k in range(1,K):
            W = square_mag(X, centroids[k-1]) #get the distances
            p = np.cumsum(W/np.sum(W)) #spread probabillities from 0 to 1
            i = p.searchsorted(np.random.rand(), 'right') #pick next center to random distance
            centroids[k,:] = X[i,:]
    elif mode=='limits':
        centroids = np.zeros((d*2,d))
        mins = np.argmin(X, axis=0)
        maxs = np.argmax(X, axis=0)
        centroids[d:] = X[mins,:]
        centroids[:d] = X[maxs,:]
    else:
        raise ValueError("Unknown mode!")
    return centroids


def kmeans(X, centroids, epsilon=0.0, max_it=1000, is_kernel=False):
    ''' kmeans(X, centroids, epsilon=0.0, max_it=1000, isKernel=False) -> yields Y, centroids, delta, step
    A generator for simple KMeans clustering.
    
    Usage:
        See demo_kmeans.py
    Args:
        X: The data N-by-d, where
            N=num datapoints
            d=dimensions
        centroids: The cluster centers.
        epsilon: Convergence threshold.
            (default=0)
        max_it: 
        isKernel: 
    Yields:
        Y: The labels or cluster association.
        centroids: The updated cluster centers.
        delta: The distance to the update step.
        step: The iteration step.
    '''
    N = X.shape[0]      #N: Size of the dataset
    K = centroids.shape[0]  #K: Number of clusters
    W = np.zeros((N,K)) #W: Distance (Weight) matrix
    C = np.zeros((N,K)) #C: Accosiation matrix
    delta = None
    
    if is_kernel:
        for k, m in enumerate(centroids):
            W[:,k] = square_mag(X, m) #calc square distances
        Y = np.argmin(W, axis=1) #find closest
        C[:,Y] = 1
    
    for step in range(max_it):
        if is_kernel:
            W = kernel_trick(X,C)
        else:
            for k, m in enumerate(centroids):
                W[:,k] = square_mag(X, m) #calc square distances
        
        Y = np.argmin(W, axis=1) #find closest
        
        tmp = centroids.copy()
        for k in range(K):
            C[:,k] = Y==k
            if any(C[:,k]):
                centroids[k,:] = np.mean(X[Y==k,:], axis=0) #calc the centroids
        
        delta = np.sum((tmp - centroids)**2) #calc the delta of change
        yield Y, centroids, delta, step #yield for mean update
        
        if delta <= epsilon:
            break #converge if change event is lower-equal than epsilon
    pass


def dbscan(X, points, radius):
    ''' dbscan(X, points, radius) -> yields Y, x, step
    Simple DBScan clustering with one path following agent.
    Takes long computation time, because almost every point needs to be checked.
    Otherwise it could not complete the pathfinding.
    The utilization of the radius is very low.
    
    Args:
        X: The dataset.
        points: Minimal number of points around core-points.
        radius: The scan radius.
    Yields:
        Y: The labels.
        x: The current scan point.
        step: The step counter.
    '''
    #guided by https://github.com/chrisjmccormick/dbscan
    N = X.shape[0]
    Y = np.zeros(N)
    radius = radius**2
    C = 1
    step=0
    
    def pick_next():
        ''' pick_next() -> yields i
        A generator to pick next unlabeled point.
        
        Yields:
            i: The next index of unlabeled datapoint.
        '''
        while True:
            i = np.nonzero(Y==0)[0]
            if i.size == 0:
                break
            else:
                yield i[0]
        pass
    
    def scan(i):
        ''' scan(i) -> yields n
        The core function of dbscan.
        Check the given point under index i to be:
        core-point, border-point, or noice.
        
        Args:
            i: The index of the point to be scanned.
        Returns:
            n: Indices of processable neighbors.
        '''
        n = square_mag(X, X[i]) <= radius
        #Do not count the current core point.
        if sum(n) > points:
            n = np.logical_and(n, np.logical_or(Y==0, Y==-1)) #check for unprocessed points
            n[i] = False
            Y[n] = C
            return np.where(n)[0]
        else:
            n = np.logical_and(n, Y==0)
            Y[n] = -1
            return np.ndarray(0)
        pass
    
    for i in pick_next():
        F = [i]
        for i in F:
            F.extend(scan(i).tolist())
            step += 1
            yield Y, X[i], step
        C += 1
    pass


LAPLACIAN_MODES = ('default', 'shi', 'jordan')
class Spectral:
    '''Spectral(X=None, gamma=1, epsilon=0, mode='default')
    Container class for spectral clustering.
    Computes the (W)eights, (D)egrees, (L)aplacian Matrix,
    the eigenvalues and eigenvectors of a given dataset (X).
    '''

    def __init__(self, X=None, gamma=1, epsilon=0, mode='default'):
        self.__gamma = float(gamma) if float(gamma) > 0 else 1
        self.__epsilon = float(epsilon)
        self.__mode = mode if mode else 'default'
        self.__N = 0
        self.__d = 0
        self.__W = 0
        self.__D = 0
        self.__L = 0
        self.__eigval = 0
        self.__eigvec = 0
        self.set(X, gamma, epsilon, mode)
        
    def set(self, X=None, gamma=None, epsilon=None, mode=None):
        ''' set(X=None, gamma=None, epsilon=None, mode=None)
        Use this function to change one or several properties.
        Recomputes the depending properties as required.
        
        Args:
            X: Set the dataset - updates the whole spectral information
            gamma: Set gamma - updates W, D, L, eigval and eigvec
            epsilon: Set epsilon - updates D, L, eigval and eigvec
            mode: Set the mode - updates L, eigval and eigvec
        '''
        recalc = False
        if isinstance(X, np.ndarray):
            self.__X = X
            self.__N = X.shape[0]
            self.__d = X.shape[1]
            recalc = True
        
        if gamma != None:
            self.__gamma = gamma
            recalc = True
        
        if recalc:
            self.__W = kernel.RBF(self.__X, self.__X, {'gamma':gamma})
        
        if epsilon != None:
            self.__epsilon = epsilon
            recalc = True
        
        if recalc:
            if self.__epsilon:
                self.__D = np.sum(self.__W > self.__epsilon, axis=1)*np.eye(self.__N)
            else:
                self.__D = np.sum(self.__W, axis=1)*np.eye(self.__N)
        
        if mode != None:
            self.__mode = mode
            recalc = True
        
        if recalc:
            L = self.__D - self.__W
            if mode == 'shi':
                D = np.sqrt(self.__D)
                D = np.linalg.inv(self.__D)
                self.__L = D*L*D
            elif mode == 'jordan':
                D = np.linalg.inv(self.__D)
                self.__L = D*L
            elif mode == 'default':
                self.__L = L
            else:
                raise ValueError("Unknown mode!")
        
            self.__eigval, self.__eigvec = np.linalg.eigh(L)
        pass
    
    def kmeans(self, K, mode='select'):
        ''' kmeans(K, mode='select') -> kmeans generator
        Initialize and returns a KMeans generator.
        (See kmeans)
        
        Args:
            K: Number of cluster centers
            mode: The initialization mode (See init_kmeans)
        Returns:
            Generator of KMeans, yields:
                Y: The labels
                means: The cluster centers
                delta: The update delta
                step: The update step counter
        '''
        X = self.__eigvec[:,:K]
        means = init_kmeans(X, K, mode)
        return kmeans(X, means)
    
    def binary(self, bits):
        if bits > 64:
            raise ValueError("bits {} > 64!".format(bits))
        C = self.__eigvec[:,1:bits+1] > 0
        Y = np.zeros(self.__N, dtype=np.int64)
        for i in range(C.shape[1]):
            Y |= np.left_shift(C[:,i], i) 
        return Y.flatten()
    
    @property
    def X(self):
        '''The inserted dataset.'''
        return self.__X
    
    @property
    def N(self):
        '''Number of datapoints.'''
        return self.__N
    
    @property
    def d(self):
        '''The dimension of the dataset.'''
        return self.__d
    
    @property
    def gamma(self):
        '''The gamma value for the RBF kernel.'''
        return self.__gamma
    
    @property
    def epsilon(self):
        '''Epsilon for knn mode.'''
        return self.__epsilon
    
    @property
    def mode(self):
        '''The KMeans initialization mode.'''
        return self.__mode
    
    @property
    def W(self):
        '''The weight matrix.'''
        return self.__W
    
    @property
    def D(self):
        '''The degree matrix.'''
        return self.__D
    
    @property
    def L(self):
        '''The laplacian matrix.'''
        return self.__L
    
    @property
    def eigval(self):
        '''The eigenvalues extracted from L.'''
        return self.__eigval
    
    @property
    def eigvec(self):
        '''The eigenvectors extracted from L.'''
        return self.__eigvec
