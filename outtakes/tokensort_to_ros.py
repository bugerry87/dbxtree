#!/usr/bin/env python

## BuildIn
from __future__ import print_function
import os
import time
from glob import glob

## Installed
import numpy as np
from scipy.spatial.transform import Rotation as R

## ROS
import rospy
import rosbag
import tf2_ros as tf
from ros_numpy import msgify
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Vector3, Point, Quaternion, TransformStamped

## Local
import mhdm.tokensort as tokensort


def decode(chunk):
	X = np.fromfile(chunk, dtype=np.uint8)
	X = tokensort.unpack_8x64(X)
	X = tokensort.realize(X)
	
	i = X[:,3]
	I = (X[:,2] & 0xF).astype(np.uint8)
	X[:,2] = X[:,2] >> 4
	X = X[:,:3].astype(np.float)
	X[:,2] -= 0x07FF
	X[:,:2] -= 0x7FFF
	X[:,:3] /= 100.0
	return X, I, i


class Token2Ros:
	node_name = 'Token2Ros'
	base_frame = 'base_link'
	map_frame = 'map'
	field_names = ('x', 'y', 'z', 'intensity')
	field_types = (np.float32,) * 3 + (np.uint8,)

	def	__init__(self,
		node_name = node_name,
		map_frame = map_frame,
		base_frame = base_frame,
		field_names = field_names,
		field_types = field_types,
		**kwargs
		):
		"""
		"""
		self.node_name = node_name
		self.map_frame = map_frame
		self.base_frame = base_frame
		self.field_names = field_names
		self.field_types = field_types
		
		self.tf = tf.TransformBroadcaster()
		self.pub_pcd = rospy.Publisher('{}/pcd'.format(self.node_name), PointCloud2, queue_size=10)
		self.pub_odom = rospy.Publisher('{}/odom'.format(self.node_name), Odometry, queue_size=10)
	
	def decode(self, path):
		odom_msg = Odometry()
		odom_msg.header.frame_id = self.map_frame
		odom_msg.child_frame_id = self.base_frame
		
		tf_msg = TransformStamped()
		tf_msg.header = odom_msg.header
		tf_msg.child_frame_id = self.base_frame
		
		odom = os.path.join(path, 'odom.bin')
		odom = np.fromfile(odom, dtype=np.float).reshape(-1,6)
		
		chunks = os.path.join(path, '??.bin')
		chunks = sorted(glob(chunks))
		
		for chunk in chunks:
			rospy.loginfo("{}: decode {}".format(args.node_name, chunk))
			X, I, i = decode(chunk)
			u = np.unique(i)
			for seq in u:
				m = i==seq
				x, y, z = X[m].T
				intens = I[m]
				
				data = np.empty(len(x), dtype=list(zip(self.field_names, self.field_types)))
				for k, v, t in zip(self.field_names, (x,y,z,intens), self.field_types):
					data[k] = v.astype(t)
				
				t = rospy.Time.now()
				pcd_msg = msgify(PointCloud2, data)
				pcd_msg.header.seq = seq
				pcd_msg.header.frame_id = self.base_frame
				pcd_msg.header.stamp = t
				
				pos = odom[seq-1,:3]
				ori = odom[seq-1,3:]
				ori = R.from_euler('zyx', ori).as_quat()
				ori = Quaternion(*ori)
				odom_msg.pose.pose.position = Point(*pos)
				odom_msg.pose.pose.orientation = ori
				odom_msg.header.seq = seq
				odom_msg.header.stamp = t
				
				tf_msg.transform.translation = Vector3(*pos)
				tf_msg.transform.rotation = ori
				print(tf_msg.header.seq)
				
				self.pub_odom.publish(odom_msg)
				self.pub_pcd.publish(pcd_msg)
				self.tf.sendTransform(tf_msg)
				yield odom_msg, pcd_msg


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
			'input_dir',
			metavar='STRING'
			)
		
		parser.add_argument(
			'--node_name', '-n',
			metavar='STRING',
			default=Token2Ros.node_name,
			help='Base name of the node.'
			)
		
		parser.add_argument(
			'--base_frame', '-b',
			metavar='STRING',
			default=Token2Ros.base_frame,
			help='The base_frame for the point cloud.'
			)
		
		parser.add_argument(
			'--map_frame', '-m',
			metavar='STRING',
			default=Token2Ros.map_frame,
			help='The map_frame for the odometry.'
			)
		
		parser.add_argument(
			'--field_names', '-f',
			metavar='STRING',
			nargs='+',
			default=Token2Ros.field_names,
			help='The field names of the PointCloud.'
			)
		
		parser.add_argument(
			'--rate', '-r',
            type=int,
            default=10,
            help="Messages per second (Herz)")
		
		parser.add_argument(
			'--clock', '-c',
			type=bool,
			nargs='?',
			help="Simulate clock",
			default=False,
			const=True)
		
		return parser
	
	args, _ = init_argparse().parse_known_args()
	rospy.init_node(args.node_name, anonymous=False, disable_signals=True)
	rospy.loginfo("Init node '{}'".format(args.node_name))
	node = Token2Ros(**args.__dict__)
	rospy.loginfo("{} ready!".format(args.node_name))
	raw_input("Press the <ENTER> to continue...")
	
	if args.clock:
		r = rospy.Rate(args.rate)
	
	for odom, pcd in node.decode(args.input_dir):
		if args.clock:
			r.sleep()
		else:
			rospy.sleep(1/args.rate)
		print(odom.pose.pose)
		
	rospy.loginfo("{} terminated!".format(args.node_name))

