

## Build In
from multiprocessing import Pool

## Installed
import numpy as np
from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Kernel import Triangle_3
from CGAL.CGAL_Kernel import Ray_3
from CGAL.CGAL_AABB_tree import AABB_tree_Triangle_3_soup

## Local
from . import spatial


class AABBTree():
	def __init__(self, X, Xi):
		X = X.astype(np.float64)
		T = [Triangle_3(*[Point_3(*p) for p in T]) for T in X[Xi]]
		self.tree = AABB_tree_Triangle_3_soup(T)
	
	def query(self, P):
		P = P.astype(np.float64)
		pp = [self.tree.closest_point_and_primitive(Point_3(*p)) for p in P]
		mp = np.array([(p[0].x(), p[0].y(), p[0].z()) for p in pp])
		nn = np.array([p[1] for p in pp])
		L = spatial.magnitude(P - mp)
		return L, mp, nn


def __job__(data):
	X, Xi, P = data
	return AABBTree(X, Xi).query(P)


def query(X, Xi, P, jobs=0):
	if jobs:
		s = int(np.ceil(len(P)/jobs))
		P = np.split(P, range(s,len(P),s))
		results = Pool(jobs).map(__job__, zip([X]*jobs, [Xi]*jobs, P))
		L = np.vstack([r[0] for r in results])
		mp = np.vstack([r[1] for r in results])
		nn = np.hstack([r[2] for r in results])
		return L, mp, nn
	else:
		return AABBTree(X, Xi).query(P)
