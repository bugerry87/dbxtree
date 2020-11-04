
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import warnings
import mhdm.bitops as bitops

y = np.fromfile('data/tokensort.bin', dtype=np.uint64)
y = y.astype(float) / y.max()
x = np.arange(len(y)).astype(float) / len(y)
coefs = []
print(y.shape)

def func(x, a, d):
	return a * np.exp(x*d)

#m, coef = curve_fit(func, x, y)
#coefs.append(coef)
#yd = y - [func(i, *m) for i in x]
#d = np.diff(yd, prepend=0)
#plt.plot(x, y, '-')
#plt.plot(x, func(x, *m), '--')
#plt.plot(x, yd)
#plt.plot(x, d)
#plt.show()

#print(np.log2(np.abs(d).min() * 2**64))
#D = (d * 2**63).astype(np.int64)
#tail = len(D) % 8
#D = D[:-tail]
#D, p = bitops.sort(D, True)
#D = bitops.transpose(D.reshape(-1,8))
#print(np.sum(D==0))
#D.tofile('data/exp_tokensort.bin')
#exit()

for o in range(1, 20):
	with warnings.catch_warnings():
		warnings.filterwarnings('error')
		try:
			coef = np.polyfit(x, y, o)
			m = np.poly1d(coef)
		except np.RankWarning:
			break
	print("R2:", r2_score(y, m(x)))
	coefs.append(coef)
	plt.plot(x, y, '-', x, m(x), '--')
	plt.show()
	y = y - m(x)
	d = np.diff(y, prepend=0)
	#plt.plot(x, y, '-', x, d, '--')
	#plt.show()
	print('Bits:', np.log2(np.abs(d).min()))
	print('Ranged:', np.abs(d).min(), np.abs(d).max())

coefs = np.hstack(coefs)
print(coefs)
