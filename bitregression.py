
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def fnc(x, a, b):
    return a * np.exp(b * x)

def dx_fnc(x, a, b):
    return a * b * np.exp(b * x)

def slop2prob(x):
    return x / 2 + 0.5

def cum_bits(fname):
    y = np.fromfile(fname, np.uint8)
    y = y[...,None] >> np.arange(8)[::-1] & 1
    y = y.reshape(-1).astype(float)
    y[y==0] = -1
    y = np.cumsum(y, dtype=float) / len(y)
    y -= y.min()
    x = np.arange(len(y), dtype=float) / len(y)
    return x, y

def cdf(x, fnc, *args):
    grad = dx_fnc(x, *args)


skip = 256
x, y = cum_bits('data/NbitTree.flg.bin')
args, covars = curve_fit(fnc, x[::skip], y[::skip])
print('Params:', args)

g = dx_fnc(0, *args)
print(slop2prob(-g), slop2prob(g))

plt.plot(x, y)
plt.plot(x, fnc(x, *args), '--')
plt.show()