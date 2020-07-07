#!/usr/bin/env python3

#build in
from threading import Thread, Condition

#installed
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial import Delaunay
from matplotlib import cm

#ros
import rospy
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point
from ros_numpy import numpify
from mesh_msgs.msg import TriangleMeshStamped, TriangleIndices

#local
import mhdm.spatial as spatial


class MeshMerge:
	def __init__(self, 
		node_name='MeshMerge', 
		topic='MeshMap/mesh', 
		frame_id='map',
		radius=0.05,
		leafsize=100,
		jobs=4
		):
		'''
		Initialize an MeshMerge node.
		Subscribes TriangleMesh cloud data.
		Publishes mesh_tool.msgs.
		
		Args:
			name [str]:	 Base name of the node.
			topic [str]:	The topic to be subscribed.
			frame_id [str]: The frame_id where the mesh came from.
		'''
		self.node_name = node_name
		self.topic = topic
		self.frame_id = frame_id
		self.radius = radius
		self.leafsize = leafsize
		self.jobs = jobs
		
		self.worker = Thread(target=self.__job__)
		self.ready = Condition()
		self.new_msg = False
		
		## init the node
		self.pub = rospy.Publisher('{}/map'.format(self.node_name), TriangleMeshStamped, queue_size=1)
		rospy.Subscriber(self.topic, TriangleMeshStamped, self.__update__, queue_size=5)
		self.worker.start()

	def __update__(self, mesh_msg):
		'''
		Update routine, to be called by subscribers.
		
		Args:
			cloud [PointCloud2]: The ROS message 'sensor_msgs.msg.PointCloud2'.
		Publishes:
			TriangleMesh
		'''
		if self.ready.acquire(False):
			self.mesh_msg = mesh_msg
			self.new_msg = True
			self.ready.notify()
			self.ready.release()
		else:
			rospy.logwarn("Node '{}' message dropped!".format(self.node_name))
		
	def __job__(self):
		first = True
		while not rospy.is_shutdown():
			self.ready.acquire()
			self.ready.wait(1.0)
			if self.new_msg:
				self.new_msg = False
			else:
				continue
			
			verts = np.array([(p.x, p.y, p.z) for p in self.mesh_msg.mesh.vertices])
			norms = np.array([(n.x, n.y, n.z) for n in self.mesh_msg.mesh.vertex_normals])
			Ti = np.array([t.vertex_indices for t in self.mesh_msg.mesh.triangles])
			
			if first:
				self.mesh = self.mesh_msg.mesh
				self.verts = verts
				self.norms = norms
				self.Ti = Ti
				first = False
				continue
			
			#Merge
			leafsize = int(self.leafsize + np.log(len(verts)) * self.leafsize)
			tree = cKDTree(self.verts, leafsize, compact_nodes=False, balanced_tree=False)
			balls = tree.query_ball_point(verts, self.radius, n_jobs=self.jobs)
			N = len(self.verts)
			
			unmerged = []
			for i, ball in enumerate(balls):
				ball_vert = np.mean(np.vstack((verts[i], self.verts[ball])), axis=0)
				ball_norm = np.mean(np.vstack((norms[i], self.norms[ball])), axis=0)
				if len(ball):
					mask = np.isin(self.Ti, ball)
					self.Ti[mask] = N + i
					self.Ti = self.Ti[mask.sum(axis=-1) <= 1]
				else:
					unmerged.append(i)
				verts[i] = ball_vert
				norms[i] = ball_norm
			
			unmerged = verts[unmerged]
			self.Ti = np.vstack((self.Ti, Ti+N))
			self.verts = np.vstack((self.verts, verts))
			self.norms = np.vstack((self.norms, norms))
			uid, idx = np.unique(self.Ti, return_inverse=True)
			srt = np.argsort(uid, axis=None)
			self.Ti = srt[idx].reshape(-1, 3)
			merged = len(self.verts) - len(uid)
			self.verts = self.verts[uid][srt]
			self.norms = self.norms[uid][srt]
			
			print("\nIncoming vertices: {}".format(len(verts)))
			print("Merged {}".format(merged))
			print("Unmerged: {}".format(len(unmerged)))
			print("Final vertices {}".format(len(self.verts)))
			
			#Publish
			mesh = self.mesh_msg.mesh
			self.mesh.triangles = [TriangleIndices(t) for t in self.Ti.tolist()]
			self.mesh.vertices = [Point(*p) for p in self.verts]
			self.mesh.vertex_normals = [Point(*n) for n in self.norms]
			self.mesh.vertex_texture_coords += mesh.vertex_texture_coords
			self.mesh.vertex_texture_coords = np.array(self.mesh.vertex_texture_coords)[uid][srt].tolist()
			self.mesh.vertex_colors += mesh.vertex_colors
			self.mesh.vertex_colors = np.array(self.mesh.vertex_colors)[uid][srt].tolist()
			
			self.mesh_msg.header.frame_id = self.frame_id
			self.mesh_msg.mesh = self.mesh
			self.pub.publish(self.mesh_msg)
		self.ready.release()


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
			default='MeshMerge',
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
			default=0.5,
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
	mesh_merge = MeshMerge(**args.__dict__)
	rospy.loginfo("Node '{}' ready!".format(args.node_name))
	rospy.spin()
	rospy.loginfo("Node '{}' terminated!".format(args.node_name))
	
