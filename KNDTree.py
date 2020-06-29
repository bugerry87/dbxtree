
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
	
	if tree.callback:
		callback = tree.callback(tree)
	
	for pi in _Pi:
		stack = deque([None])
		stack.append(root.query(tree, pi))
		node = stack.pop()
		while node:
			for n in node:
				if n:
					stack.append(n)
			if tree.callback:
				next(callback)
			node = stack.pop()
	return tree


class K3DTree:
	class Leaf:
		def __init__(self, tree, Ti):
			self.Ti = Ti
			self.leaf_size = len(Ti)
		
		def __len__(self):
			return self.leaf_size
		
		def __str__(self):
			return str(len(self))
		
		def query(self, tree, Pi):
			def query_point(X, Xi, XP, Pi):
				L = spatial.magnitude(XP).min(axis=-1)
				Lmin = L < tree.L[Pi]
				if np.any(Lmin):
					Pi = Pi[Lmin]
					tree.L[Pi] = L[Lmin]
					tree.nn[Pi] = -1 
					tree.nn[Pi,0] = Xi[Lmin]
					tree.mp[Pi] = X[Lmin]
					tree.done[self.Ti] = True
			
			def query_line(Xi, Pi, mp):
				L = spatial.magnitude(mp)
				L = L.min(axis=-1)
				Lmin = L < tree.L[Pi]
				if np.any(Lmin):
					Pi = Pi[Lmin]
					tree.L[Pi] = L[Lmin]
					tree.nn[Pi] = -1 
					tree.nn[Pi,:2] = Xi
					tree.mp[Pi] = tree.P[Pi,0] + mp[Lmin]
					tree.done[self.Ti] = True
					
			def query_face(Xi, Pi, mp):
				L = spatial.magnitude(mp - tree.P[Pi])
				L = L.min(axis=-1)
				Lmin = L < tree.L[Pi]
				if np.any(Lmin):
					Pi = Pi[Lmin]
					tree.L[Pi] = L[Lmin]
					tree.nn[Pi] = Xi
					tree.mp[Pi] = mp[Lmin]
					tree.done[self.Ti] = True
				pass
			
			for Ti in self.Ti:
				X = tree.T[Ti]
				Xi = tree.Xi[Ti]
				XP = tree.P[Pi] - X
				a = np.sum(XP * tree.x[Ti], axis=-1) / tree.m[Ti].flatten()
				
				is_line = (a > 0) & (a < 1)
				point = np.nonzero(~is_line)
				line = np.nonzero(is_line)
				
				if point[0].size:
					k, n = point
					query_point(X[n], Xi[n], XP[point], Pi[k])
				
				if line[0].size:
					k, n = line
					mp = tree.x[Ti][n] * a[line][:,None] - XP[line]
					query_line(Xi[:2], Pi[k], mp)
				else:
					continue
					
				if tree.N < 3:
					continue
				
				a = np.sum(XP[k] * tree.eN[Ti], axis=-1) / tree.eM[Ti]
				face = np.nonzero(np.all(a <= 0, axis=-1))
				if face[0].size:
					k, n = face
					mp = np.mean(mp[k] + tree.eN[Ti] * a[k], axis=1)
					query_face(Xi, Pi[k], mp)
			pass

	class Node:
		def __init__(self, Ti, depth):
			self.Ti = Ti
			self.depth = depth
			self.left = None
			self.center = None
			self.right = None
			self.__expanded = False
		
		def __len__(self):
			return len(self.Ti)
		
		def __str__(self):
			return "{:-<4}-+---Left:{}" \
				"\n          {}  \\_Center:{}" \
				"\n          {}  \\__Right:{}".format(
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
			
			T = tree.T[self.Ti]
			self.mean = T.reshape(-1, tree.D).mean(axis=0)
			T = T - self.mean
			
			self.mag = 0.0
			while self.mag == 0.0:
				if tree.N > 2:
					k = np.random.choice(len(T))
					n = np.random.choice(tree.N)
					self.norm = tree.eN[self.Ti[k],n].flatten()
				else:
					self.norm = np.random.randn(tree.D)
				self.mag = spatial.magnitude(self.norm)
			a = np.sum(np.dot(T, self.norm) > 0.0, axis=-1)
			
			left = self.Ti[a==0]
			center = self.Ti[a==1]
			right = self.Ti[a==2]
			
			if self.left:
				pass
			elif len(left) > tree.leaf_size:
				self.left = K3DTree.Node(left, self.depth+1)
			elif len(left):
				self.left = K3DTree.Leaf(tree, left)
			
			if self.center:
				pass
			elif len(center) > tree.leaf_size:
				self.center = K3DTree.Node(center, self.depth+1)
			elif len(center):
				self.center = K3DTree.Leaf(tree, center)
			
			if self.right:
				pass
			elif len(right) > tree.leaf_size:
				self.right = K3DTree.Node(right, self.depth+1)
			elif len(right):
				self.right = K3DTree.Leaf(tree, right)
		
		def query(self, tree, Pi):
			self.__expand__(tree)
			a = np.dot(tree.P[Pi,0] - self.mean, self.norm) / self.mag
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
		self.Xi = Xi
		self.T = X[Xi]
		self.K, self.N, self.D = self.T.shape
		self.x = self.T[:,(*range(1,self.N),0)] - self.T
		self.m = spatial.magnitude(self.x)
		self.m[self.m==0] = 1 #fix for zero div
		
		if self.N > 2:
			x = self.x.reshape(-1,self.D)
			fN = np.cross(x, -self.x[:,::-1].reshape(-1,self.D))
			self.eN = np.cross(x, fN).reshape(-1,self.N,self.D)
			self.eM = spatial.magnitude(self.eN)
			self.fN = fN.reshape(-1,self.N,self.D)
			
		self.roots = [K3DTree.Node(Ti, 0) for Ti in np.array_split(np.arange(len(Xi)), j)]
		self.batch_size = batch_size
		self.callback = callback
		self.leaf_size = leaf_size if leaf_size else 1 + self.K / 100
		self.done = np.zeros(self.K, dtype=bool)
	
	def __str__(self):
		return "**KDNtree**\n  Leaf Size: {}\n".format(self.leaf_size) + \
			"\n".join(["  Root:{}".format(str(root)) for root in self.roots])
	
	def __len__(self):
		return self.K
	
	def query(self, P):
		j = len(self.roots)
		self.P = P.reshape(-1,1,self.D)
		self.mp = np.zeros(P.shape)
		self.nn = -np.ones((len(P), self.N), dtype=int)
		self.L = np.zeros(len(P)) + np.inf
		self.Pi = np.arange(len(P))
		
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


def callback(tree):
	last = 0
	while last <= 50:
		curr = int(tree.done.mean() * 50)
		dif = curr - last
		if curr > last:
			print('#' * dif, end='', flush=True)
		last = curr
		yield

###TEST
if __name__ == '__main__':
	from argparse import ArgumentParser
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	from utils import time_delta
	from time import time
	
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
			default=60
			)
		
		parser.add_argument(
			'--query_size', '-q',
			metavar='INT',
			type=int,
			default=60
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
			default=10
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
	
	np.random.seed(args.seed)
	X = np.random.randn(args.model_size,3)
	P = np.random.randn(args.query_size,3)
	Xi = np.arange(len(X)).reshape(-1,3)
	X[Xi[:,1:]] *= 0.5
	X[Xi[:,1:]] += X[Xi[:,0]]
	
	print("K3DTree")
	print("Model size:", X.shape)
	print("Query size:", P.shape)
	
	delta = time_delta(time())
	tree = K3DTree(
		X, Xi,
		args.jobs,
		args.leaf_size,
		args.batch_size,
		callback=callback
		)
	
	print("\n0%					  |50%					 |100%")
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
	ax.set_zlim3d(-2.4, 2.4)
	plt.show()
	print(tree)