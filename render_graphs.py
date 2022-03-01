
import numpy as np
import matplotlib.pyplot as plt

bpp_avg = np.loadtxt('data/training/run-20220105-144844_test-tag-epoch_bpp.csv', delimiter=',', skiprows=1)
bpp_min = np.loadtxt('data/training/run-20220105-144844_test-tag-epoch_bpp_min.csv', delimiter=',', skiprows=1)
bpp_max = np.loadtxt('data/training/run-20220105-144844_test-tag-epoch_bpp_max.csv', delimiter=',', skiprows=1)

plt.title('Bpp-ratio per Epoch on Testset')

t, x, y = bpp_max.T
plt.plot(x+1, y, 'b.--')

t, x, y = bpp_avg.T
plt.plot(x+1, y, 'b.-')

t, x, y = bpp_min.T
plt.plot(x+1, y, 'b.--')
plt.ylabel('bpp (min, average, max)')

t, x, y = bpp_avg.T
plt.twinx()
plt.plot(x+1, y, 'c--')
plt.ylabel('bpp (average)')

table = plt.table(cellText=[
		[f'{x:1.2f}' for x in bpp_max.T[-1]],
		[f'{x:1.2f}' for x in bpp_avg.T[-1]],
		[f'{x:1.2f}' for x in bpp_min.T[-1]]
	],
	rowLabels=['max', 'avg', 'min'],
	colLabels=[f'epoch {x:1.0f}' for x in range(1,len(t)+1)],
	loc='bottom')

plt.xticks([])
plt.tight_layout()
plt.show()
plt.clf()

#-----------------------

train_loss = np.loadtxt('data/training/run-20220105-144844_train-tag-epoch_loss.csv', delimiter=',', skiprows=1)
train_acc = np.loadtxt('data/training/run-20220105-144844_train-tag-epoch_accuracy.csv', delimiter=',', skiprows=1)
val_loss = np.loadtxt('data/training/run-20220105-144844_validation-tag-epoch_loss.csv', delimiter=',', skiprows=1)
val_acc = np.loadtxt('data/training/run-20220105-144844_validation-tag-epoch_accuracy.csv', delimiter=',', skiprows=1)

plt.title('Training & Validation')

t, x, y = train_loss.T
plt.plot(x+1, y, 'b.-', label='train loss')
t, x, y = val_loss.T
plt.plot(x+1, y, 'b.--', label='val loss')
plt.ylabel('loss')
plt.legend(loc='upper left')

ax = plt.twinx()
t, x, y = train_acc.T
plt.plot(x+1, y, 'c.-', label='train acc')
t, x, y = val_acc.T
plt.plot(x+1, y, 'c.--', label='val acc')
plt.ylabel('accuracy')
plt.legend(loc='upper right')

table = plt.table(cellText=[
		[f'{x:1.4f}' for x in train_loss.T[-1]],
		[f'{x:1.4f}' for x in val_loss.T[-1]],
		[f'{x:1.4f}' for x in train_acc.T[-1]],
		[f'{x:1.4f}' for x in val_acc.T[-1]]
	],
	rowLabels=['train loss', 'val loss', 'train acc', 'val acc'],
	colLabels=[f'epoch {x:1.0f}' for x in range(1,len(t)+1)],
	loc='bottom')

plt.xticks([])
plt.tight_layout()
plt.show()
plt.clf()