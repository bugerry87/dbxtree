"""
Operations for LiDAR data.

Author: Gerald Baulig
"""

# Installed
import numpy as np

# Local
from . import spatial


def psnr(A, B=None, peek=100):
	if B is None:
		MSE = A
	else:
		MSE = np.mean((A - B) ** 2)
	if MSE == 0:
		return 100
	else:
		return 20 * np.log10(peek / np.sqrt(MSE))


def xyz2uvd(X, norm=False, z_off=0.0, d_off=0.0, mode='sphere'):
	x, y, z = X.T
	pi = np.where(x > 0.0, np.pi, -np.pi)
	uvd = np.empty(X.shape)
	with np.errstate(divide='ignore', over='ignore'):
		uvd[:,0] = np.arctan(x / y) + (y < 0) * pi
		uvd[:,2] = np.linalg.norm(X, axis=-1)
		if mode == 'sphere':
			uvd[:,1] = np.arcsin((z + z_off) / (uvd[:,2] + d_off))
		elif mode == 'cone':
			uvd[:,1] = (z + z_off) / (np.linalg.norm(X[:,:2], axis=-1) + d_off)
		else:
			raise ValueError("Unknown mode: '{}'!".format(mode))
	if norm is False:
		pass
	elif norm is True:
		uvd = spatial.prob(uvd)
	else:
		uvd[:,norm] = spatial.prob(uvd[:,norm])
	return uvd


def uvd2xyz(U, z_off=0.0, d_off=0.0, mode='sphere'):
	u, v, d = U.T
	xyz = np.empty(U.shape)
	xyz[:,0] = np.sin(u) * (d + z_off) 
	xyz[:,1] = np.cos(u) * (d + z_off) 
	if mode == 'sphere':
		xyz[:,2] = np.sin(v) * (d + d_off) - z_off
	elif mode == 'cone':
		xyz[:,2] = v * (d + d_off) + z_off # * (np.linalg.norm(xyz[:,:2], axis=-1) + r_off) + z_off
	else:
		raise ValueError("Unknown mode: '{}'!".format(mode))
	return xyz


def dot_keypoints(X, m=1, o=0):
	assert(m > 0)
	assert(o >= 0)
	mask = np.zeros(len(X), dtype=bool)
	P = iter(X)
	k = next(P)
	p0 = next(P)
	p0k = p0 - k
	p0km = spatial.magnitude(p0k)
	mag = p0km
	m = m**2
	mask[0] = True
	mask[-1] = True
	
	for i, p1 in enumerate(P):
		pp = p1 - p0
		ppm = spatial.magnitude(pp)
		mag += ppm
		
		p1k = p1 - k
		p1km = spatial.magnitude(p1k)
		dot = np.dot(p0k, p1k)**2 / (p0km * p1km)
		
		if dot < 1 - np.exp(-o-mag/m)**4:
			#new keypoint detected
			k = p0
			p0 = p1
			p0k = pp
			p0km = ppm
			mag = ppm
			mask[i+1] = True
		else:
			#update
			p0 = p1
			p0k = p1k
			p0km = p1km
	return X[mask], mask


def kalman_filter(Z, order=2, R=0.03, dt=1.0):
	from filterpy.common import kinematic_kf
	x_dim = Z.shape[-1]
	z_dim = Z.shape[-1]
	kf = kinematic_kf(x_dim, z_dim, dt, order, order_by_dim=False)
	X = np.empty((len(Z), len(kf.F)))
	p = np.empty((len(Z), len(kf.F)))
	
	for i, z in enumerate(Z):
		kf.predict()
		kf.update(z.reshape(-1,1))
		X[i] = kf.x.flatten()
		p[i] = kf.x_prior.flatten()
	return X, p
