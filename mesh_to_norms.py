#!/usr/bin/env python3

#Installed
import numpy as np

#ros
import rospy
from ros_numpy import msgify
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point
from mesh_msgs.msg import TriangleMeshStamped


class Mesh2Norms:
	node_name = 'Mesh2Norms'
	topic = '/MeshGen/mesh'
	field_names = ('x', 'y', 'z', 'xn', 'yn', 'zn', 'face')
	field_types = (np.float32,) * 6 + (np.byte,)

	def __init__(self, 
		node_name = node_name, 
		topic = topic,
		field_names = field_names,
		field_types = field_types
		):
		'''
		Initialize an Mesh2Norms node.
		Subscribes TriangleMesh cloud data.
		Publishes PointCloud2.
		
		Args:
			name [str]:	 Base name of the node.
			topic [str]: The topic to be subscribed.
		'''
		self.node_name = node_name
		self.topic = topic
		self.field_names = field_names
		self.field_types = field_types
		
		self.mesh_map = TriangleMeshStamped()
		self.map = []
		self.header = None
		
		## init the node
		self.pub = rospy.Publisher('{}/norms'.format(self.node_name), PointCloud2, queue_size=10)
		rospy.Subscriber(self.topic, TriangleMeshStamped, self.__update__, queue_size=10)

	def __update__(self, mesh_msg):
		'''
		Update routine, to be called by subscribers.
		
		Args:
			mesh_msg [TriangleMeshStamped]
		Publishes:
			PointCloud2
		'''
		verts = mesh_msg.mesh.vertices
		norms = mesh_msg.mesh.vertex_normals
		
		V = np.array([(p.x, p.y, p.z, n.x, n.y, n.z, 0) for p, n in zip(verts, norms)])
		T = V[np.array([t.vertex_indices for t in mesh_msg.mesh.triangles], dtype=int), :3]
		F = np.hstack((np.mean(T, axis=1), np.cross(T[:,1] - T[:,0], T[:,2] - T[:,0]), np.ones((len(T),1))))
		frame = np.vstack((V, F))
		
		data = np.empty(len(frame), dtype=list(zip(self.field_names, self.field_types)))
		for k, v, t in zip(self.field_names, frame.T, self.field_types):
			data[k] = v.astype(t)
		
		cloud_msg = msgify(PointCloud2, data)
		cloud_msg.header = mesh_msg.header
		self.header = mesh_msg.header
		self.pub.publish(cloud_msg)
		print("Publish {}: Vertex normals {}".format(cloud_msg.header.stamp, data.shape))
		pass


if __name__ == '__main__':
	from argparse import ArgumentParser

	def init_argparse(parents=[]):
		'''
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
			default=Mesh2Norms.node_name,
			help='Base name of the node.'
			)
		
		parser.add_argument(
			'--topic', '-t',
			metavar='TOPIC',
			default=Mesh2Norms.topic,
			help='The topic to be subscribed.'
			)
		
		parser.add_argument(
			'--field_names', '-f',
			nargs=7,
			metavar='STRING',
			default=Mesh2Norms.field_names,
			help='The field names of the Point Cloud.'
			)
		return parser


	args, _ = init_argparse().parse_known_args()
	rospy.init_node(args.node_name, anonymous=False)
	rospy.loginfo("Init node '{}' on topic '{}'".format(args.node_name, args.topic))
	node = Mesh2Norms(**args.__dict__)
	rospy.loginfo("Node '{}' ready!".format(args.node_name))
	rospy.spin()
	rospy.loginfo("Node '{}' terminated!".format(args.node_name))
	
