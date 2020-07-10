#!/usr/bin/env python3

#build in
from threading import Thread, Condition

#installed
import numpy as np
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


def numpy_to_trianglemesh(verts, trids, norms, uvds=None, colors=None):
	mesh = TriangleMeshStamped()
	mesh.mesh.vertices = [Point(*p) for p in verts]
	mesh.mesh.vertex_normals = [Point(*n) for n in norms]
	mesh.mesh.triangles = [TriangleIndices(t) for t in trids.tolist()]
	if uvds is not None:
		mesh.mesh.vertex_texture_coords = [Point(*uvd) for uvd in uvds]
	if colors is not None:
		mesh.mesh.vertex_colors = [ColorRGBA(*colors) for colors in colors]
	return mesh


class MeshGen:
	def __init__(self, 
		node_name='MeshGen', 
		topic='~/velodyne_points', 
		frame_id='base_link', 
		fields=('x','y','z','intensity'), 
		colors='Spectral'
		):
		'''
		Initialize an MeshGen node.
		Subscribes velodyne_point cloud data.
		Publishes mesh_tool.msgs.
		
		Args:
			name [str]:	 Base name of the node.
			topic [str]:	The topic to be subscribed.
			frame_id [str]: The frame_id in rviz where the markers get plotted at.
		'''
		self.node_name = node_name
		self.topic = topic
		self.frame_id = frame_id
		self.fields = fields
		self.colors = cm.get_cmap(colors)
		self.worker = Thread(target=self.__job__)
		self.ready = Condition()
		self.new_msg = False
		
		self.rec_msg = 0
		self.rec_pts = 0
		self.pro_msg = 0
		self.pro_pts = 0
		
		## init the node
		self.pub = rospy.Publisher('{}/mesh'.format(self.node_name), TriangleMeshStamped, queue_size=10)
		rospy.Subscriber(self.topic, PointCloud2, self.__update__, queue_size=10)
		self.worker.start()

	def __update__(self, cloud_msg):
		'''
		Update routine, to be called by subscribers.
		
		Args:
			cloud [PointCloud2]: The ROS message 'sensor_msgs.msg.PointCloud2'.
		Publishes:
			TriangleMesh
		'''
		self.rec_msg += 1
		self.rec_pts += cloud_msg.width
		
		if self.ready.acquire(False):
			self.cloud_msg = cloud_msg
			self.new_msg = True
			self.ready.notify()
			self.ready.release()
		else:
			#rospy.logwarn("{} message dropped!".format(self.node_name))
			pass
		
	def __job__(self):
		while not rospy.is_shutdown():
			self.ready.acquire()
			self.ready.wait(1.0)
			if self.new_msg:
				self.new_msg = False
			else:
				continue

			self.pro_msg += 1
			self.pro_pts += self.cloud_msg.width
			
			X = numpify(self.cloud_msg)
			x, y, z, i = self.fields
			Q = np.array((X[x], X[y], X[z])).T
			P = spatial.sphere_uvd(Q)
			
					   
			surf = Delaunay(P[:,(0,1)])
			Ti = surf.simplices
			fN = spatial.face_normals(Q[Ti])
			vN = spatial.vec_normals(fN, Ti.flatten())

			mesh = numpy_to_trianglemesh(Q, Ti, vN, P, self.colors(X[i]))
			mesh.header = self.cloud_msg.header
			mesh.header.frame_id = self.frame_id
			self.pub.publish(mesh)
	
 
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
			default='MeshGen',
			help='Base name of the node.'
			)
		
		parser.add_argument(
			'--topic', '-t',
			metavar='TOPIC',
			default='~/velodyne_points',
			help='The topic to be subscribed.'
			)
		
		parser.add_argument(
			'--frame_id', '-f',
			metavar='STRING',
			default='base_link',
			help='The frame_id for rviz to plot the markers at.'
			)
		
		parser.add_argument(
			'--fields', '-F',
			metavar='STRING',
			nargs='+',
			default=('x', 'y', 'z', 'intensity'),
			help='The field names of the PointCloud.'
			)
		
		parser.add_argument(
			'--colors', '-c',
			metavar='STRING',
			default='Spectral',
			help='The color mapping method.'
			)
		
		return parser


	args, _ = init_argparse().parse_known_args()
	rospy.init_node(args.node_name, anonymous=False)
	rospy.loginfo("Init node '{}' on topic '{}'".format(args.node_name, args.topic))
	node = MeshGen(**args.__dict__)
	rospy.loginfo("{} ready!".format(args.node_name))
	rospy.spin()
	rospy.loginfo("\n".join((
		"\n+{:-^32}+".format(args.node_name),
		"|received messages:  {:>12}|".format(node.rec_msg),
		"|received points:    {:>12}|".format(node.rec_pts),
		"|processed messages: {:>12}|".format(node.pro_msg),
		"|processed points:   {:>12}|".format(node.pro_pts),
		"+{:-^32}+".format("")
		)))
	rospy.loginfo("{} terminated!".format(args.node_name))
	
