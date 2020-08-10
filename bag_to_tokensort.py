#!/usr/bin/env python

## BuildIn
from __future__ import print_function
import os

## Installed
import numpy as np

## ROS
import rosbag
from ros_numpy import numpify
from sensor_msgs.msg import PointCloud2

## Local
import mhdm.tokensort as tokensort


def bag_to_numpy(bag, topic, fields):
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
		X = np.array((X[x], X[y], X[z], X[i], np.full(N, seq, dtype=np.float))).T
		X[:,3] /= X[:,3].max()
		X[:,3] *= 0xF
		X[:,:3] *= 100
		X[:,2] += 2**11
		X[:,:2] += np.iinfo(np.uint16).max * 0.5
		X = np.round(X).astype(np.uint16)
		yield X


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
			'--topic', '-t',
			metavar='TOPIC',
			default='/MapPCD/pcd',
			help='The topic to be subscribed.'
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
		return parser
	
	
	def alloc_file(prefix, suffix):
		fn = os.path.join(prefix, '0x{:0>2}.bin'.format(hex(suffix)[2:]))
		return open(fn, 'wb')
	
	args, _ = init_argparse().parse_known_args()
		
	print("\nLoad data: {} - topic: {}".format(args.bag, args.topic))
	print("--- Build Chunks ---")
	
	try:
		files = {}
		for X in bag_to_numpy(args.bag, args.topic, args.fields):
			X = tokensort.pack_64(X)
			X = tokensort.featurize(X)
			X.sort()
			X = np.ndarray((len(X),8), dtype=np.uint8, buffer=X)
			u, i = np.unique(X[:,-1], return_index=True)
			I = np.roll(i+1, -1)
			I[-1] = len(X)
			
			for u, i, I in zip(u, i, I):
				if u in files:
					fid = files[u]
				else:
					fid = alloc_file(args.output_dir, u)
					files[u] = fid
				X[i:I, :-1].tofile(fid)
				print("Add to file: {} {:>8} Bytes".format(fid.name, (I-i)*7))
	finally:
		for fid in files.values():
			fid.close()
	
	print("--- Sort Chunks ---")
		
	for fid in files.values():
		X = np.fromfile(fid.name, dtype=np.uint8).reshape(-1,7)
		X = np.hstack((X, np.zeros((len(X),1), dtype=np.uint8)))
		X = np.ndarray(len(X), dtype=np.uint64, buffer=X)
		X.sort()
		X = tokensort.numeric_delta(X)
		X = tokensort.pack_8x64(X).T
		X[:-1].tofile(fid.name)
		print("File sorted:", fid.name)

