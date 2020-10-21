
#BuildIn
from argparse import ArgumentParser
from multiprocessing import Pool, Lock
from collections import deque
import time

#Installed
import numpy as np

#Local
import spatial


def init_MeshTree_args(parents=[]):
	"""
	Initialize an ArgumentParser for this module.
	
	Args:
		parents: A list of ArgumentParsers of other scripts, if there are any.
		
	Returns:
		parser: The ArgumentParsers.
	"""
	parser = ArgumentParser(
		#description="Demo for embedding data via LDA",
		parents=parents
		)
	
	parser.add_argument(
		'--model_size', '-m',
		metavar='INT',
		type=int,
		default=30000
		)
	
	parser.add_argument(
		'--query_size', '-q',
		metavar='INT',
		type=int,
		default=30000
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
		default=1000
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


def __job__(data):
	np.random.seed(0)
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
				tree.nn[Pi,:2] = Xi[Lmin]
				tree.mp[Pi] = tree.P[Pi,0] + mp[Lmin]
				tree.done[self.Ti] = True
				
		def query_face(Xi, Pi, mp):
			L = spatial.magnitude(mp)
			L = L.min(axis=-1)
			Lmin = L < tree.L[Pi]
			if np.any(Lmin):
				Pi = Pi[Lmin]
				tree.L[Pi] = L[Lmin]
				tree.nn[Pi] = Xi
				tree.mp[Pi] = tree.P[Pi,0] + mp[Lmin]
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
				n = (n, (n + 1) % tree.N)
				query_line(Xi[n,].T, Pi[k], mp)
			else:
				continue
				
			if tree.N < 3:
				continue
			
			ui, un = np.unique(k, return_counts=True)
			ui = ui[un==tree.N]
			if not len(ui):
				continue
			
			XP = XP[ui]
			a = np.sum(XP * tree.eN[Ti], axis=-1) / tree.eM[Ti].flatten()
			face = np.all(a <= 0, axis=-1)
			if np.any(face):
				a = np.sum(XP[face] * tree.fN[Ti], axis=-1) / tree.fM[Ti].flatten()
				mp = np.mean(tree.fN[Ti] * a[:,None], axis=1)
				query_face(Xi, Pi[ui][face], -mp)
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
				k = np.random.choice(len(T)) #Better choice than random?
				n = np.random.choice(tree.N)
				self.norm = tree.eN[self.Ti[k],n].flatten()
			else:
				self.norm = np.random.randn(tree.D)
			self.mag = spatial.magnitude(self.norm)
			
		a = np.sum(np.dot(T, self.norm) > 0.0, axis=-1)
		left = self.Ti[a==0]
		center = self.Ti[(a > 0) & (a < tree.D)]
		right = self.Ti[a==tree.D]
		
		if self.left:
			pass
		elif len(left) > tree.leaf_size:
			self.left = Node(left, self.depth+1)
		elif len(left):
			self.left = Leaf(tree, left)
		
		if self.center:
			pass
		elif len(center) > tree.leaf_size:
			self.center = Node(center, self.depth+1)
		elif len(center):
			self.center = Leaf(tree, center)
		
		if self.right:
			pass
		elif len(right) > tree.leaf_size:
			self.right = Node(right, self.depth+1)
		elif len(right):
			self.right = Leaf(tree, right)
	
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


class MeshTree:
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
			fN = np.cross(x, -self.x[:,range(-1,self.N-1)].reshape(-1,self.D))
			self.eN = np.cross(x, fN).reshape(-1,self.N,self.D)
			self.eM = spatial.magnitude(self.eN)
			self.fN = fN.reshape(-1,self.N,self.D)
			self.fM = spatial.magnitude(self.fN)
			
		self.roots = [Node(Ti, 0) for Ti in np.array_split(np.arange(len(Xi)), j)]
		self.batch_size = batch_size
		self.callback = callback
		self.leaf_size = leaf_size if leaf_size else 1 + self.K / 100
		self.done = np.zeros(self.K, dtype=bool)
	
	def __str__(self):
		return "**MeshTree**\n  Leaf Size: {}\n".format(self.leaf_size) + \
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


def query():
	pass


def demo(args):
	pass


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
	from mpl_toolkits.mplot3d.art3d import Poly3DCollection
	from utils import time_delta
	from time import time
	
	
	
	args, _ = init_argparse().parse_known_args()
	
	np.random.seed(args.seed)
	X = np.random.randn(args.model_size,3)
	P = np.random.randn(args.query_size,3)
	Xi = np.arange(len(X)).reshape(-1,3)
	
	print("MeshTree")
	print("Model size:", X.shape)
	print("Query size:", P.shape)
	
	delta = time_delta(time())
	tree = MeshTree(
		X, Xi,
		args.jobs,
		args.leaf_size,
		args.batch_size,
		callback=callback
		)
	
	print("\n0%					  |50%					 |100%")
	dist, mp, nn = tree.query(P)
	
	print("\nQuery time:", next(delta))
	print("Mean loss:", np.sqrt(dist).mean())
	
	fig = plt.figure()
	ax = fig.add_subplot((111), projection='3d')
	mp -= P
	typ = np.sum(nn == -1, axis=-1)
	point = typ == 2
	line = typ == 1
	face = typ == 0
	
	poly = Poly3DCollection(X)
	poly.set_alpha(0.5)
	poly.set_edgecolor('b')
	ax.add_collection3d(poly)
	ax.scatter(*P.T, color='r')
	ax.quiver(*P[point].T, *mp[point].T, color='g')
	ax.quiver(*P[line].T, *mp[line].T, color='y')
	ax.quiver(*P[face].T, *mp[face].T, color='b')
	
	ax.set_xlim3d(-3, 3)
	ax.set_ylim3d(-3, 3)
	ax.set_zlim3d(-2.4, 2.4)
	plt.show()
	print(tree)