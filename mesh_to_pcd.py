#!/usr/bin/env python3

#Installed
import numpy as np

#ros
import rospy
from ros_numpy import msgify
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point
from mesh_msgs.msg import TriangleMeshStamped


class Mesh2PCD:
	def __init__(self, 
		node_name='Mesh2PCD', 
		topic='/MeshMap/mesh'
		):
		'''
		Initialize an Mesh2PCD node.
		Subscribes TriangleMesh cloud data.
		Publishes PointCloud2.
		
		Args:
			name [str]:	 Base name of the node.
			topic [str]: The topic to be subscribed.
		'''
		self.node_name = node_name
		self.topic = topic
		self.mesh_map = TriangleMeshStamped()
		self.map = []
		self.header = None
		
		## init the node
		self.pub_frame = rospy.Publisher('{}/frame'.format(self.node_name), PointCloud2, queue_size=5)
		self.pub_map = rospy.Publisher('{}/map'.format(self.node_name), PointCloud2, queue_size=1)
		rospy.Subscriber(self.topic, TriangleMeshStamped, self.__update__, queue_size=5)

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
		names = ('x', 'y', 'z', 'xn', 'yn', 'zn')
		data = np.empty(len(verts), dtype=list(zip(names, (np.float32,) * 6)))
		frame = [(p.x, p.y, p.z, n.x, n.y, n.z) for p, n in zip(verts, norms)]
		self.map += frame
		
		for k, v in zip(names, np.array(frame).T):
			data[k] = v
		
		cloud_msg = msgify(PointCloud2, data)
		cloud_msg.header = mesh_msg.header
		self.header = mesh_msg.header
		print("Publish {}: Frame size {}, Map size: {}".format(cloud_msg.header.stamp, data.shape, len(self.map)))
		self.pub_frame.publish(cloud_msg)
		pass
	
	def publish_map(self):
		names = ('x', 'y', 'z', 'xn', 'yn', 'zn')
		data = np.empty(len(self.map), dtype=list(zip(names, (np.float32,) * 6)))
		
		for k, v in zip(names, np.array(self.map).T):
			data[k] = v
		
		cloud_msg = msgify(PointCloud2, data)
		cloud_msg.header.seq = 1
		cloud_msg.header.frame_id = 'map'
		print("Publish Map: PointCloud2 {}".format(data.shape))
		self.pub_map.publish(cloud_msg)
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
			default='Mesh2PCD',
			help='Base name of the node.'
			)
		
		parser.add_argument(
			'--topic', '-t',
			metavar='TOPIC',
			default='/MeshMap/mesh',
			help='The topic to be subscribed.'
			)
		
		return parser


	args, _ = init_argparse().parse_known_args()
	rospy.init_node(args.node_name, anonymous=False)
	rospy.loginfo("Init node '{}' on topic '{}'".format(args.node_name, args.topic))
	node = Mesh2PCD(**args.__dict__)
	rospy.loginfo("Node '{}' ready!".format(args.node_name))
	input("Press any key to publish map!")
	node.publish_map()
	input("Press any key to quit!")
	rospy.loginfo("Node '{}' terminated!".format(args.node_name))
	
