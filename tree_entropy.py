#!/usr/bin/env python3

## Installed
import numpy as np

X = np.fromfile('/share/token/00_bak.bin', dtype=np.uint64)
X.sort()
X = np.ndarray((len(X)//3,3,2), np.dtype=np.uint32, buffer=X)

class Node:
	def __init__(self):
		pass
	
	def expand(self):
		pass

class TokenTree:
	
