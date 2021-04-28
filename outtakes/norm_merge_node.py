#!/usr/bin/env python

#installed
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial import Delaunay

#ros
import rospy
from ros_numpy import msgify
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point
from mesh_msgs.msg import TriangleMeshStamped, TriangleIndices


class NormMerge:
	def __init__(self, 
		node_name='NormMerge', 
		topic='MeshMap/mesh', 
		frame_id='map',
		radius=0.05,
		leafsize=100,
		jobs=4
		):
		self.node_name = node_name
		self.topic = topic
		self.map_id = frame_id
		self.radius = radius
		self.leafsize = leafsize
		self.jobs = jobs
		self.first = True
		self.names = ('x', 'y', 'z', 'xn', 'yn', 'zn')
		
		## init the node
		self.pub = rospy.Publisher('{}/norms'.format(self.node_name), PointCloud2, queue_size=1)
		rospy.Subscriber(self.topic, TriangleMeshStamped, self.__update__, queue_size=20)
		pass

	def __update__(self, mesh_msg):
		v = mesh_msg.mesh.vertices
		vn = mesh_msg.mesh.vertex_normals
		frame = np.array([(p.x, p.y, p.z, n.x, n.y, n.z) for p, n in zip(v, vn)], dtype=np.float32)
		keep = np.zeros(len(frame), dtype=bool)
		
		if self.first:
			self.map = frame
			self.keep = keep
			self.first = False
			return
		
		N = len(self.map)
		leafsize = int(self.leafsize + np.log(N) * self.leafsize)
		tree = cKDTree(self.map[:,:3], leafsize, compact_nodes=False, balanced_tree=False)
		balls = tree.query_ball_point(frame[:,:3], self.radius, n_jobs=self.jobs)
		
		for i, ball in enumerate(balls):
			ball_vert = np.mean(np.vstack((frame[i,:3], self.map[ball,:3])), axis=0)
			ball_norm = np.mean(np.vstack((frame[i,3:], self.map[ball,3:])), axis=0)
			frame[i,:3] = ball_vert
			frame[i,3:] = ball_norm
			if len(ball):
				self.keep[ball] = False
				keep[i] = True
		
		self.map = np.vstack((self.map[self.keep], frame))
		print("\nIncoming vertices: {}".format(len(frame)))
		print("Merged vertices: {}".format(np.sum(~self.keep)))
		print("Final vertices {}".format(len(self.map)))
		self.keep = np.hstack((self.keep[self.keep], keep))
		
		#Publish
		data = np.empty(len(self.map), dtype=list(zip(self.names, (np.float32,) * 6)))
		for k, v in zip(self.names, np.array(self.map).T):
			data[k] = v
		
		self.cloud_msg = msgify(PointCloud2, data)
		self.cloud_msg.header = mesh_msg.header
		self.pub.publish(self.cloud_msg)
		pass


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
			description="ROS node to transform PointCloud2 to TriangleMash.",
			parents=parents
			)
		
		parser.add_argument(
			'--node_name', '-n',
			metavar='STRING',
			default='NormMerge',
			help='Base name of the node.'
			)
		
		parser.add_argument(
			'--topic', '-t',
			metavar='TOPIC',
			default='/MeshMap/mesh',
			help='The topic to be subscribed.'
			)
		
		parser.add_argument(
			'--frame_id', '-f',
			metavar='STRING',
			default='map',
			help='The frame id where the map came is to plot at.'
			)
		
		parser.add_argument(
			'--radius', '-r',
			type=float,
			metavar='FLOAT',
			default=1.0,
			help='The merge radius for each point.'
			)
		
		parser.add_argument(
			'--leafsize', '-l',
			type=int,
			metavar='INT',
			default=100,
			help='Minimal leaf size for KDTree.'
			)
		
		parser.add_argument(
			'--jobs', '-j',
			type=int,
			metavar='INT',
			default=4,
			help='How many jobs for KDTree.'
			)
		
		return parser


	args, _ = init_argparse().parse_known_args()
	rospy.init_node(args.node_name, anonymous=False)
	rospy.loginfo("Init node '{}' on topic '{}'".format(args.node_name, args.topic))
	node = NormMerge(**args.__dict__)
	rospy.loginfo("Node '{}' ready!".format(args.node_name))
	
	while not rospy.is_shutdown():
		opt = raw_input("(q)uit or (p)ublish map: ")
		if opt is 'q':
			break
		elif opt is not 'p':
			continue
		elif not node.first:
			node.pub.publish(node.cloud_msg)
		
	rospy.loginfo("Node '{}' terminated!".format(args.node_name))
	
