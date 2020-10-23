"""
Operations for LiDAR data.

Author: Gerald Baulig
"""

# Installed
import numpy as np

# Local
from . import spatial


def yield_keypoints_xyz(P, m=1, yield_offsets=False):
	P = iter(P)
	k = next(P)
	p0 = next(P)
	p0k = p0 - k
	p0km = spatial.magnitude(p0k)
	mag = p0km
	m = m**2
	
	for i, p1 in enumerate(P):
		pp = p1 - p0
		ppm = spatial.magnitude(pp)
		mag += ppm
		
		p1k = p1 - k
		p1km = spatial.magnitude(p1k)
		dot = np.dot(p0k, p1k)**2 / (p0km * p1km) 
		
		if dot < 1 - np.exp(-mag/m)**4:
			#new keypoint detected
			k = p0
			p0 = p1
			p0k = pp
			p0km = ppm
			mag = ppm
			if yield_offsets:
				yield k, k * 0
			else:
				yield k
		else:
			#update
			p0 = p1
			p0k = p1k
			p0km = p1km
			if yield_offsets:
				yield k, p1 - k
	pass


def mask_keypoints_xyz(P, m=1):
	mask = np.zeros(len(P), dtype=bool)
	O = P.copy()
	P = iter(P)
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
		
		if dot < 1 - np.exp(-mag/m)**4:
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
			O[i+1] = p1k
	return O, mask


def mask_keypoints_uvd(P, t):
	n = len(P)
	mask = np.zeros(n, dtype=bool)
	mask[0] = True
	O = P.copy()
	P = iter(P)
	k = next(P)
	
	for i, p in zip(range(1,n),P):
		delta = p[-1] - k[-1]
		m = np.abs(delta) > t  
		mask[i] = m 
		if m:
			k = p
		else:
			O[i,-1] = delta
	return O, mask


