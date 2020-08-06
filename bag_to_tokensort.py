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


def bag_to_numpy(bag, topic, fields, chunk_size=None):
	bag = rosbag.Bag(bag).read_messages(topics=[topic])

	def chunk(msg):
		n = 0
		while msg and (chunk_size is None or n < chunk_size):
			pc = PointCloud2()
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
			n += 1
			msg = next(bag, [None]*3)[1]
			yield X
	
	msg = next(bag, [None]*3)[1]
	while msg:
		X = np.hstack([x for x in chunk(msg)]).T
		X[:,3] /= X[:,3].max()
		X[:,3] *= 0xF
		X[:,:3] *= 100
		X[:,2] += 2**11
		X[:,:2] += np.iinfo(np.uint16).max * 0.5
		X = np.round(X).astype(np.uint16)
		msg = next(bag, [None]*3)[1]
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
			default='/share/tokensort.bin'
			)
		
		parser.add_argument(
			'--visualize', '-V',
			metavar='STRING',
			nargs='*',
			default=[],
			choices=('cloud', 'dist')
			)	
		return parser
	
	args, _ = init_argparse().parse_known_args()
		
	print("\nLoad data: {} - topic: {}".format(args.bag, args.topic))
	for i, X in enumerate(bag_to_numpy(args.bag, args.topic, args.fields, 1000)):
		X = tokensort.pack_64(X)
		fn = '{}_org_{:0>4}.bin'.format(args.output_name.replace('.bin', ''), i)
		X.tofile(fn)
		Y = tokensort.encode(X)
		fn = '{}_{:0>4}.bin'.format(args.output_name.replace('.bin', ''), i)
		Y.tofile(fn)
		print("Save to file:", fn)


	if 'cloud' in args.visualize:
		import mhdm.viz as viz
		fig = viz.create_figure()
		I = X[:,2] & 0xF
		X[:,2] = X[:,2] >> 4
		viz.vertices(X, I, fig, None)
		viz.show_figure()
	
	if 'dist' in args.visualize:
		import matplotlib.pyplot as plt
		import matplotlib.ticker as ticker
	
		@ticker.FuncFormatter
		def major_formatter(i, pos):
			return "{:0>8}".format(bin(int(i))[2:])
		
		Y = Y.flatten()[::10]
		ax = plt.subplot(111)
		ax.set_ylim(-7, 263)
		ax.yaxis.set_major_formatter(major_formatter)
		ax.scatter(range(len(Y)), Y, s=0.2, marker='.')
		plt.show()
