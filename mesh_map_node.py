#!/usr/bin/env python

#installed
from scipy.spatial.transform import Rotation as R

#ros
import rospy
import tf
from geometry_msgs.msg import Point
from mesh_msgs.msg import TriangleMeshStamped


class MeshMap:
	def __init__(self, 
		name='MeshMap', 
		topic='~/velodyne_points', 
		base_link='base_link', 
		frame_id='map'
		):
		'''
		Initialize an MeshMap node.
		Subscribes TriangleMesh cloud data.
		Publishes mesh_tool.msgs.
		
		Args:
			name [str]:	 Base name of the node.
			topic [str]:	The topic to be subscribed.
			frame_id [str]: The frame_id where the mesh came from.
		'''
		self.name = name
		self.topic = topic
		self.base_link = base_link
		self.frame_id = frame_id
		self.listener = tf.TransformListener()
		
		## init the node
		self.pub = rospy.Publisher('{}/mesh'.format(self.name), TriangleMeshStamped, queue_size=5)
		rospy.Subscriber(self.topic, TriangleMeshStamped, self.__update__, queue_size=5)

	def __update__(self, mesh_msg):
		'''
		Update routine, to be called by subscribers.
		
		Args:
			mesh_msg [TriangleMeshStamped]
		Publishes:
			TriangleMeshStamped
		'''
		#try:
		trans, quat = self.listener.lookupTransform(self.frame_id, self.base_link, rospy.Time(0))
		#except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
		#	print(e)
		#	return
		
		#Decode
		verts = mesh_msg.mesh.vertices
		norms = mesh_msg.mesh.vertex_normals
		verts = [(p.x, p.y, p.z) for p in verts]
		norms = [(p.x, p.y, p.z) for p in norms]
		
		#Transform
		rot = R.from_quat(quat)
		verts = rot.apply(verts)
		norms = rot.apply(norms)
		verts += trans
		
		#Encode
		mesh_msg.mesh.vertices = [Point(*p) for p in verts]
		mesh_msg.mesh.vertex_normals = [Point(*n) for n in norms]
		mesh_msg.header.frame_id = self.frame_id
		
		self.pub.publish(mesh_msg)


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
			'--base_name', '-n',
			metavar='STRING',
			default='MeshMap',
			help='Base name of the node.'
			)
		
		parser.add_argument(
			'--topic', '-t',
			metavar='TOPIC',
			default='/MeshGen/mesh',
			help='The topic to be subscribed.'
			)
		
		parser.add_argument(
			'--frame_id', '-f',
			metavar='STRING',
			default='map',
			help='The frame id where the map came is to plot at.'
			)
		
		parser.add_argument(
			'--base_link', '-b',
			metavar='STRING',
			default='base_link',
			help='The transformer of the car.'
			)
		
		return parser


	args, _ = init_argparse().parse_known_args()
	rospy.init_node(args.base_name, anonymous=False)
	rospy.loginfo("Init node '{}' on topic '{}'".format(args.base_name, args.topic))
	mesh_merge = MeshMap(args.base_name, args.topic, args.base_link, args.frame_id)
	rospy.loginfo("Node '{}' ready!".format(args.base_name))
	rospy.spin()
	
