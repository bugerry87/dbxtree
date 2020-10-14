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
	X = X[np.abs(X[:,2]) <= 0x07FF]
	X[:,2] += 0x07FF
	X[:,:2] += 0x7FFF
	
	X = np.round(X).astype(np.uint16)
	X = tokensort.pack_64(X)
	return tokensort.featurize(X)


def odom_to_numpy(bag, topic):
	for _, odom, _ in rosbag.Bag(bag).read_messages(topics=[topic]):
		p = odom.pose.pose.position
		p = np.array((p.x, p.y, p.z))
		o = odom.pose.pose.orientation
		o = R.from_quat((o.x, o.y, o.z, o.w))
		o = o.as_euler('zyx')
		yield np.hstack((p, o))


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
		return parser
	
	
	def alloc_file(prefix, suffix):
		fn = os.path.join(prefix, '{:0>2}.bin'.format(suffix))
		return open(fn, 'wb')
	
	args, _ = init_argparse().parse_known_args()
	
	if args.odom:	
		print("\nLoad data: {} - topic: {}".format(args.bag, args.odom))
		print("--- Convert Odometry ---")
		
		fn = os.path.join(args.output_dir, 'odom.bin')
		X = odom_to_numpy(args.bag, args.odom)
		np.array([x for x in X]).tofile(fn)
			
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

