#!/usr/bin/env python

## BuildIn
from __future__ import print_function

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
		X = np.array((X[x], X[y], X[z], X[i], np.full(N, seq, dtype=np.float)))
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
			'--output_name', '-y',
			metavar='PATH',
			default='/share/token.bin'
			)
		
		parser.add_argument(
			'--visualize', '-V',
			metavar='STRING',
			nargs='*',
			default=[],
			choices=('cloud', 'dist')
			)	
		return parser
	
	
	def alloc_files(prefix):
		for surfix in range(256):
			fn = '{}_0x{:0>2}.bin'.format(prefix, hex(surfix)[2:])
			yield open(fn, 'wb')
	
	args, _ = init_argparse().parse_known_args()
	args.output_name = args.output_name.replace('.bin', '')
		
	print("\nLoad data: {} - topic: {}".format(args.bag, args.topic))
	
	try:
		files = [f for f in alloc_files(args.output_name)]
		for X in bag_to_numpy(args.bag, args.topic, args.fields):
			X = tokensort.pack_64(X)
			X = tokensort.featurize(X)
			X = np.ndarray((len(X),8), dtype=np.uint8, buffer=X)
			
			for surfix, file in enumerate(files):
				m = X[:,-1] == surfix
				X[m, :-1].tofile(file)
				print(len(m)*7, "bytes add to file:", fn)
	finally:
		for f in files:
			f.close()
		
	for surfix, file in enumerate(files):
		X = np.fromfile(file.name, dtype=np.uint8).reshape(-1,7)
		X = np.hstack((X, np.zeros((len(X),1))))
		X = np.ndarray(len(X), dtype=np.uint64, buffer=X)
		X.sort()
		X = tokensort.numeric_delta(X)
		X = tokensort.pack_8x64(X).T
		X[:-1].tofile(file.name)
		print("File sorted:", fn)
