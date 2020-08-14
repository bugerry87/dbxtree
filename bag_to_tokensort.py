#!/usr/bin/env python

## BuildIn
from __future__ import print_function
import os

## Installed
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R

## ROS
import rosbag
from ros_numpy import numpify
from sensor_msgs.msg import PointCloud2

## Local
import mhdm.tokensort as tokensort
import mhdm.spatial as spatial


def pc_to_numpy(bag, topic, fields):
	pc = PointCloud2()
	
	for _, msg, _ in rosbag.Bag(bag).read_messages(topics=[topic]):
		pc.header = msg.header
		pc.height = msg.height
		pc.width = msg.width
		pc.fields = msg.fields
		pc.is_bigendian = msg.is_bigendian
		pc.point_step = msg.point_step
		pc.row_step = msg.row_step
		pc.data = msg.data
		pc.is_dense = msg.is_dense
		
		seq = pc.header.seq
		N = pc.width
		
		x, y, z, i = fields
		X = numpify(pc)
		yield np.array((X[x], X[y], X[z], X[i], np.full(N, seq, dtype=np.float))).T


def planarize(X, dot=0.9):
	P = spatial.sphere_uvd(X[:,(1,0,2)])
	mesh = Delaunay(P[:,(0,1)])
	Ti = mesh.simplices
	fN = spatial.face_normals(X[Ti,:3])
	vN = spatial.vec_normals(fN, Ti.flatten())
	m = spatial.mask_planar(vN, fN, Ti.flatten(), dot)
	return X[m], m


def featurize(X):
	X[:,3] /= X[:,3].max()
	X[:,3] *= 0xF
	X[:,:3] *= 100
	X[:,2] += 2**11
	X[:,:2] += np.iinfo(np.uint16).max * 0.5
	
	X = np.round(X).astype(np.uint16)
	X = tokensort.pack_64(X)
	return tokensort.featurize(X)


def odom_to_numpy(bag, topic, d=180, b=7, rec_err=None):
	abs_pos = np.zeros(3)
	abs_ori = R.from_rotvec((0,0,0))
	n = float(np.iinfo(np.int8).max)
	
	if rec_err is not None:
		rec_err['seq'] = []
		rec_err['pos_err'] = []
		rec_err['ori_err'] = []
		cum_pos = np.zeros(3)
		cum_ori = R.from_rotvec((0,0,0))
	
	for _, odom, _ in rosbag.Bag(bag).read_messages(topics=[topic]):
		p = odom.pose.pose.position
		p = (p.x, p.y, p.z)
		delta_p = p - abs_pos
		a = np.where(delta_p >= 0, 100, -100)
		delta_p = np.round(a * np.abs(delta_p)**(1.0/b)).astype(np.int8)
		abs_pos[:] = p
		
		o = odom.pose.pose.orientation
		o = R.from_quat((o.x, o.y, o.z, o.w))
		delta_o = (o * abs_ori.inv()).as_euler('zyx', degrees=True)
		a = np.where(delta_p >= 0, n, -n)
		delta_o = a * np.abs(delta_o/d)**(1.0/b)
		delta_o = np.round(delta_o).astype(np.int8)
		abs_ori = o
		
		idx = np.array(odom.header.seq, dtype=np.uint16)
		idx = np.ndarray(2, np.int8, buffer=idx)
		
		if rec_err is not None:
			cum_pos += (delta_p/100.0)**b
			pos_err = abs_pos - cum_pos
			
			ori_err = d * (delta_o/n)**b
			cum_ori *= R.from_euler('zyx', ori_err, degrees=True)
			ori_err = (cum_ori * abs_ori.inv()).as_euler('zyx', degrees=True)
			
			rec_err['seq'].append(odom.header.seq)
			rec_err['pos_err'].append(pos_err)
			rec_err['ori_err'].append(ori_err)
			
			print("\nSeq [{:>5}] ".format(odom.header.seq),
				"\nPos {:<16}".format(delta_p),
				"Error", pos_err,
				"\nOri {:<16}".format(delta_o),
				"Error", ori_err)
		yield np.hstack((delta_p, delta_o, idx))


if __name__ == '__main__':
	from argparse import ArgumentParser
	
	def init_argparse(parents=[]):
		''' init_argparse(parents=[]) -> parser
		Initialize an ArgumentParser for this module.
		
		Args:
			parents: A list of ArgumentParsers of other scripts, if there are any.
			
		Returns:
			parser: The ArgumentParsers.
		'''
		parser = ArgumentParser(
			description="Converts BAG to TokenSort.bin",
			parents=parents
			)
		
		parser.add_argument(
			'bag',
			metavar='STRING'
			)
		
		parser.add_argument(
			'--points', '-p',
			metavar='TOPIC',
			default='/MapPCD/pcd',
			help='The topic of point clouds.'
			)
		
		parser.add_argument(
			'--odom', '-o',
			metavar='TOPIC',
			default='/MapPCD/odom',
			help='The topic of odometries.'
			)
		
		parser.add_argument(
			'--fields', '-f',
			metavar='STRING',
			nargs='+',
			default=('x', 'y', 'z', 'intensity'),
			help='The field names of the PointCloud.'
			)
		
		parser.add_argument(
			'--output_dir', '-y',
			metavar='PATH',
			default='/share/token/'
			)
		
		parser.add_argument(
			'--visualize', '-V',
			metavar='STRING',
			nargs='*',
			default=[],
			choices=('cloud', 'dist')
			)
		
		parser.add_argument(
			'--planarize', '-P',
			metavar='FLOAT',
			type=float,
			default=0.0
			)
		
		parser.add_argument(
			'--chunk_size', '-c',
			metavar='INT',
			type=int,
			default=1024
			)
		
		parser.add_argument(
			'--ang_range', '-d',
			metavar='FLOAT',
			type=float,
			default=180.0
			)
		
		parser.add_argument(
			'--ang_scale', '-b',
			metavar='INT',
			type=int,
			default=7
			)
		
		return parser
	
	
	def alloc_file(prefix, suffix):
		fn = os.path.join(prefix, '~{:0>2}.tmp'.format(suffix))
		return open(fn, 'wb')
	
	args, _ = init_argparse().parse_known_args()
	
	if args.odom:	
		print("\nLoad data: {} - topic: {}".format(args.bag, args.odom))
		print("--- Convert Odometry ---")
		
		rec_err = {}
		fn = os.path.join(args.output_dir, 'odom.bin')
		X = odom_to_numpy(args.bag, args.odom, args.ang_range, args.ang_scale, rec_err)
		tokensort.encode(np.array([x for x in X])).tofile(fn)
		
		import matplotlib.pyplot as plt
		seq = rec_err['seq']
		pos_err = np.array(rec_err['pos_err']).T
		for err, label in zip(pos_err, 'xyz'):
			plt.plot(seq, err, label=label)
		plt.legend()
		plt.show()
		exit()
			
	if args.points:
		print("\nLoad data: {} - topic: {}".format(args.bag, args.points))
		print("--- Build Chunks ---")
		
		try:
			files = []
			fid = alloc_file(args.output_dir, len(files))
			for i, X in enumerate(pc_to_numpy(args.bag, args.points, args.fields)):
				if args.planarize:
					X, m = planarize(X, args.planarize)
					print(i, " - Planarize point drop:", len(m), "-->", m.sum())
					
				X = featurize(X)
				
				if not i % args.chunk_size:
					if fid:
						fid.close()
					fid = alloc_file(args.output_dir, len(files))
					files.append(fid)
				
				X.tofile(fid)
				print("Add to file: {} {:>8} Bytes".format(fid.name, X.size * 8))
		finally:
			if fid:
				fid.close()
			
		print("--- Sort n Add Chunks ---")
		for fid in files:
			X = np.fromfile(fid.name, dtype=np.uint64)
			X.sort()
			X = tokensort.pack_8x64(X).T
			X.tofile(fid.name)
			print("File sorted:", fid.name, X.shape)

