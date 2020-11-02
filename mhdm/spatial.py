"""
Spatial operations for 3D.

Author: Gerald Baulig
"""

# Installed
import numpy as np
from scipy.spatial import cKDTree


def magnitude(X, sqrt=False):
	if len(X.shape) == 1:
		m = np.sum(X**2)
	else:
		shape = X.shape[:-1] + (1,)
		m = np.sum(X**2, axis=-1).reshape(*shape)
	return np.sqrt(m) if sqrt else m


def norm(X, magnitude=False):
	if len(X.shape) == 1:
		m = np.linalg.norm(X)
	else:
		shape = X.shape[:-1] + (1,)
		m = np.linalg.norm(X, axis=-1).reshape(*shape)
	n = X / m
	if magnitude:
		return n, m
	else:
		return n


def prob(X):
	X = X.copy()
	X -= X.min(axis=0)
	X /= X.max(axis=0)
	return X


def face_normals(T, normalize=True, magnitude=False):
	fN = np.cross(T[:,1] - T[:,0], T[:,2] - T[:,0])
	if normalize:
		return norm(fN, magnitude)
	else:
		return fN


def edge_normals(T, fN=None, normalize=True, magnitude=False):
	if fN is None:
		fN = face_normals(T, False)
	fN = fN.repeat(3, axis=0)
	xN = T[:,(1,2,0)] - T
	xN = xN.reshape(-1,3)
	eN = np.cross(xN, fN).reshape(-1,3,3)
	if normalize:
		return norm(eN, magnitude)
	else:
		return eN


def vec_normals(fN, Ti, normalize=True, magnitude=False):
	Ti = Ti.flatten()
	fN = fN.repeat(3, axis=0)
	vN = np.zeros((Ti.max()+1, 3))
	for fn, i in zip(fN, Ti):
		vN[i] += fn
	if normalize:
		return norm(vN, magnitude)
	else:
		return vN


def face_magnitude(T=None, fN=None, normalize=False):
	if fN is None:
		fN = face_normals(T, False)
	mag = magnitude(fN, normalize)
	if normalize:
		return mag / 2
	else:
		return mag


def nn_point2point(X, P):
	return cKDTree(X).query(P)


def nn_point2line(X, Xi, P):
	nn = -np.ones((P.shape[0],2), dtype=int)
	dist, nn[:,0] = nn_point2point(X, P)
	dist = dist**2
	mp = X[nn[:,0]]
	A = X[Xi[:,0]]
	B = X[Xi[:,1]]
	AB = B - A
	ABm = magnitude(AB)
	for i, p in enumerate(P):
		Ap = p - A
		a = np.sum(AB * Ap, axis=1) / ABm.flatten()
		m = ((a > 0) * (a < 1)).astype(bool)
		if any(m):
			ap = AB[m] * a[m][:,None] - Ap[m]
			L = magnitude(ap)
			Larg = np.argmin(L)
			Lmin = L[Larg]
			if Lmin < dist[i]:
				nn[i] = Xi[m][Larg]
				dist[i] = Lmin
				mp[i] = p + ap[Larg]
	return dist, mp, nn


def raycast(T, rays, fN=None, eN=None, back=False):
	r = rays[:,1] - rays[:,0]
	if fN is None:
		fN = face_normals(T, True)
	if eN is None:
		eN = edge_normals(T, fN, False)
	
	idx = []
	intrp = []
	hit = np.zeros(len(T), dtype=bool)
	for (a, b), r in zip(rays, r):
		an = (a - T[:,0]) * fN
		am = np.sum(an, axis=-1)
		bm = np.sum((b - T[:,0]) * fN, axis=-1)
		on_plane = ((am >= 0) & (bm <= 0)) | (back & ((am <= 0) & (bm >= 0)))
		
		am = am[on_plane]
		fn = fN[on_plane]
		en = eN[on_plane]
		t = T[on_plane]
		
		m = np.sqrt(am).reshape(-1,1) / magnitude(r * fn, True)
		mp = r.reshape(1,-1) * m + a
		mp = mp.repeat(3, axis=0).reshape(-1,3,3)
		in_trid = np.all(np.sum((mp - t) * en, axis=-1) <= 0, axis=-1)
		hit[on_plane] |= in_trid
		idx.append(np.nonzero(in_trid)[0])
		intrp.append(mp[in_trid, 0])
	return hit, idx, intrp


def mask_planar(vN, fN, Ti, min_dot=0.9, mask=None):
	Ti = Ti.flatten()
	fN = fN.repeat(3, axis=0)
	if mask is None:
		mask = np.ones(Ti.max()+1, dtype=bool)
	for fn, i in zip(fN, Ti):
		if mask[i]:
			mask[i] &= np.dot(vN[i], fn) <= min_dot
		else:
			pass
	return mask

