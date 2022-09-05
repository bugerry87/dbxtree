#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

bpp_avg = np.loadtxt('data/training/run-20220105-144844_test-tag-epoch_bpp.csv', delimiter=',', skiprows=1)
bpp_min = np.loadtxt('data/training/run-20220105-144844_test-tag-epoch_bpp_min.csv', delimiter=',', skiprows=1)
bpp_max = np.loadtxt('data/training/run-20220105-144844_test-tag-epoch_bpp_max.csv', delimiter=',', skiprows=1)

plt.figure(figsize=(5,4))
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

#-----------------------

train_loss = np.loadtxt('data/training/run-20220105-144844_train-tag-epoch_loss.csv', delimiter=',', skiprows=1)
train_acc = np.loadtxt('data/training/run-20220105-144844_train-tag-epoch_accuracy.csv', delimiter=',', skiprows=1)
val_loss = np.loadtxt('data/training/run-20220105-144844_validation-tag-epoch_loss.csv', delimiter=',', skiprows=1)
val_acc = np.loadtxt('data/training/run-20220105-144844_validation-tag-epoch_accuracy.csv', delimiter=',', skiprows=1)

plt.figure(figsize=(5,4))
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

#-----------------------
'''
uids_bpp = np.loadtxt('data/training/run-r3.0_k128_uids_test-tag-epoch_bpp.csv', delimiter=',', skiprows=1)
pivots_bpp = np.loadtxt('data/training/run-r3.0_k128_pivots_test-tag-epoch_bpp.csv', delimiter=',', skiprows=1)
pos_bpp = np.loadtxt('data/training/run-r3.0_k128_pos_test-tag-epoch_bpp.csv', delimiter=',', skiprows=1)
all_bpp = np.loadtxt('data/training/run-r3.0_k128_all_fixed_test-tag-epoch_bpp.csv', delimiter=',', skiprows=1)

plt.figure(figsize=(5,4))
plt.title('Feature Evaluation')

t, x, y = uids_bpp.T
plt.plot(x+1, y, label='NIDs')
t, x, y = pos_bpp.T
plt.plot(x+1, y, label='Position')
t, x, y = pivots_bpp.T
plt.plot(x+1, y, label='Trestles')
t, x, y = all_bpp.T
plt.plot(x+1, y, label='All')
plt.ylabel('bpp')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
'''
#-----------------------

dbx = pd.read_csv('data/training/DBXTreeEarlyStop_KITTI.csv', header=0)
mpge = pd.read_csv('data/training/MPEGbaselineKITTI.csv', header=0)
draco = pd.read_csv('data/training/DracoBaselineKITTI.csv', header=0)

plt.figure(figsize=(5,4))
plt.title('KITTI raw 2011_09_26_drive_0005')

x, y = draco.loc[:,['bpp','rmsF-p2point']].values.T
plt.plot(x, y, label='DRACO', marker='o')

x, y = mpge.loc[:,['bpp','rmsF-p2point']].values.T
plt.plot(x, y, label='MPEG G-PCC', marker='^')

x, y = dbx.query('Precision == "r0003"').loc[:,['bpp','rmsF-p2point']].values.T
plt.plot(x, y, label=r'DBX $\tau$=0.3cm', marker='.')

x, y = dbx.query('Precision == "r0006"').loc[:,['bpp','rmsF-p2point']].values.T
plt.plot(x, y, label=r'DBX $\tau$=0.6cm', marker='v')

x, y = dbx.query('Precision == "r0010"').loc[:,['bpp','rmsF-p2point']].values.T
plt.plot(x, y, label=r'DBX $\tau$=1.0cm', marker='x')

plt.grid(True)
plt.xlim((0,15))
plt.ylim((0,0.07))
plt.xlabel('bpp (input)')
plt.ylabel('Chamfer Distance (m)')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

#-----------------------

plt.figure(figsize=(5,4))
plt.title('KITTI raw 2011_09_26_drive_0005')

x, y = draco.loc[:,['bpp','rmsFPSNR-p2point']].values.T
plt.plot(x, y, label='DRACO', marker='o')

x, y = mpge.loc[:,['bpp','rmsFPSNR-p2point']].values.T
plt.plot(x, y, label='MPEG G-PCC', marker='^')

x, y = dbx.query('Precision == "r0003"').loc[:,['bpp','rmsFPSNR-p2point']].values.T
plt.plot(x, y, label=r'DBX $\tau$=0.3cm', marker='.')

x, y = dbx.query('Precision == "r0006"').loc[:,['bpp','rmsFPSNR-p2point']].values.T
plt.plot(x, y, label=r'DBX $\tau$=0.6cm', marker='v')

x, y = dbx.query('Precision == "r0010"').loc[:,['bpp','rmsFPSNR-p2point']].values.T
plt.plot(x, y, label=r'DBX $\tau$=1.0cm', marker='x')

plt.grid(True)
plt.xlim((0,15))
plt.ylim((40,75))
plt.xlabel('bpp (input)')
plt.ylabel('P2Point-PSNR (p=8.7m)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

#-----------------------

plt.figure(figsize=(5,4))
plt.title('KITTI raw 2011_09_26_drive_0005')

x, y = draco.loc[:,['bpp','rmsFPSNR-p2plane']].values.T
plt.plot(x, y, label='DRACO', marker='o')

x, y = mpge.loc[:,['bpp','rmsFPSNR-p2plane']].values.T
plt.plot(x, y, label='MPEG G-PCC', marker='^')

x, y = dbx.query('Precision == "r0003"').loc[:,['bpp','rmsFPSNR-p2plane']].values.T
plt.plot(x, y, label=r'DBX $\tau$=0.3cm', marker='.')

x, y = dbx.query('Precision == "r0006"').loc[:,['bpp','rmsFPSNR-p2plane']].values.T
plt.plot(x, y, label=r'DBX $\tau$=0.6cm', marker='v')

x, y = dbx.query('Precision == "r0010"').loc[:,['bpp','rmsFPSNR-p2plane']].values.T
plt.plot(x, y, label=r'DBX $\tau$=1.0cm', marker='x')

plt.grid(True)
plt.xlim((0,15))
plt.ylim((45,80))
plt.xlabel('bpp (input)')
plt.ylabel('P2Plane-PSNR (p=8.7m)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

#-----------------------

dbx = pd.read_csv('data/training/DBXTreeEarlyStop_FORD.csv', header=0)
mpge = pd.read_csv('data/training/MPEGbaselineFORD.csv', header=0)
sparse = pd.read_csv('data/training/SparseVoxelsFORD.csv', header=0)

plt.figure(figsize=(5,4))
plt.title('Ford (subset)')

x, y = mpge.loc[:,['bpp','rmsF-p2point']].values.T
plt.plot(x, y/1000, label='MPEG G-PCC', marker='^')

x, y = sparse.loc[:,['bpp','rmsF-p2point']].values.T
plt.plot(x, y/1000, label='SparsePCGC', marker='o')

x, y = dbx.query('Precision == "r0001"').loc[:,['bpp','rmsF-p2point']].values.T
plt.plot(x, y, label=r'DBX $\tau$=0.1cm', marker='.')

x, y = dbx.query('Precision == "r0003"').loc[:,['bpp','rmsF-p2point']].values.T
plt.plot(x, y, label=r'DBX $\tau$=0.3cm', marker='v')

x, y = dbx.query('Precision == "r0006"').loc[:,['bpp','rmsF-p2point']].values.T
plt.plot(x, y, label=r'DBX $\tau$=0.6cm', marker='x')

plt.grid(True)
#plt.xlim((0,15))
#plt.ylim((0,0.07))
plt.xlabel('bpp (input)')
plt.ylabel('Chamfer Distance (m)')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

plt.figure(figsize=(5,4))
plt.title('Ford (subset)')

x, y = mpge.loc[:,['bpp','rmsFPSNR-p2point']].values.T
plt.plot(x, y, label='MPEG G-PCC', marker='^')

x, y = sparse.loc[:,['bpp','rmsFPSNR-p2point']].values.T
plt.plot(x, y, label='SparsePCGC', marker='o')

x, y = dbx.query('Precision == "r0001"').loc[:,['bpp','rmsFPSNR-p2point']].values.T
plt.plot(x, y, label=r'DBX $\tau$=0.1cm', marker='.')

x, y = dbx.query('Precision == "r0003"').loc[:,['bpp','rmsFPSNR-p2point']].values.T
plt.plot(x, y, label=r'DBX $\tau$=0.3cm', marker='v')

#x, y = dbx.query('Precision == "r0006"').loc[:,['bpp','rmsFPSNR-p2point']].values.T
#plt.plot(x, y, label=r'DBX $\tau$=0.6cm', marker='x')

plt.grid(True)
#plt.xlim((0,15))
#plt.ylim((45,80))
plt.xlabel('bpp (input)')
plt.ylabel('P2Point-PSNR (p=18m)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

#-----------------------

plt.figure(figsize=(5,4))
plt.title('Ford (subset)')

x, y = mpge.loc[:,['bpp','rmsFPSNR-p2plane']].values.T
plt.plot(x, y, label='MPEG G-PCC', marker='^')

x, y = sparse.loc[:,['bpp','rmsFPSNR-p2plane']].values.T
plt.plot(x, y, label='SparsePCGC', marker='o')

x, y = dbx.query('Precision == "r0001"').loc[:,['bpp','rmsFPSNR-p2plane']].values.T
plt.plot(x, y, label=r'DBX $\tau$=0.1cm', marker='.')

x, y = dbx.query('Precision == "r0003"').loc[:,['bpp','rmsFPSNR-p2plane']].values.T
plt.plot(x, y, label=r'DBX $\tau$=0.3cm', marker='v')

#x, y = dbx.query('Precision == "r0006"').loc[:,['bpp','rmsFPSNR-p2plane']].values.T
#plt.plot(x, y, label=r'DBX $\tau$=0.6cm', marker='x')

plt.grid(True)
#plt.xlim((0,15))
#plt.ylim((45,80))
plt.xlabel('bpp (input)')
plt.ylabel('P2Plane-PSNR (p=18m)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()