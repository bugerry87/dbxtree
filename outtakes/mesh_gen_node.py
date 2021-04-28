#!/usr/bin/env python2


## build in
from threading import Thread, Condition

## installed
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R
from matplotlib import cm

## ros
import rospy
import tf
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point
from ros_numpy import numpify
from mesh_msgs.msg import TriangleMeshStamped, TriangleIndices


def magnitude(X, sqrt=False):
	if len(X.shape) == 1:
		m = np.sum(X**2)
	else:
		m = np.sum(X**2, axis=-1).reshape(*(X.shape[:-1] + (1,)))
	return np.sqrt(m) if sqrt else m


def norm(X, magnitude=False):
	if len(X.shape) == 1:
		m = np.linalg.norm(X)
	else:
		m = np.linalg.norm(X, axis=-1).reshape(*(X.shape[:-1] + (1,)))
	n = X / m
	if magnitude:
		return n, m
	else:
		return n


def prob(X):
	X = X.copy()
	X -= X.min(axis=0)
	X /= X.max(axis=0)
	return X


def sphere_uvd(X, norm=False, z_off=0.0, r_off=0.0):
	x, y, z = X.T
	pi = np.where(x > 0.0, np.pi, -np.pi)
	uvd = np.empty(X.shape)
	with np.errstate(divide='ignore', over='ignore'):
		uvd[:,0] = np.arctan(x / y) + (y < 0) * pi
		uvd[:,2] = np.linalg.norm(X, axis=-1)
		uvd[:,1] = np.arcsin((z-z_off) / uvd[:,2]-r_off)
	
	if norm is False:
		pass
	elif norm is True:
		uvd = prob(uvd)
	else:
		uvd[:,norm] = prob(uvd[:,norm])
	return uvd


def face_normals(T, normalize=True, magnitude=False):
	fN = np.cross(T[:,1] - T[:,0], T[:,2] - T[:,0])
	if normalize:
		return norm(fN, magnitude)
	else:
		return fN


def vec_normals(fN, Ti_flat, normalize=True, magnitude=False):
	fN = fN.repeat(3, axis=0)
	vN = np.zeros((Ti_flat.max()+1, 3))
	for fn, i in zip(fN, Ti_flat):
		vN[i] += fn
	if normalize:
		return norm(vN, magnitude)
	else:
		return vN


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
	## defaults
	node_name = 'MeshGen'
	topic = '~/velodyne_points'
	base_frame = 'base_link'
	map_frame = 'map'
	min_distance = 1.0
	fields = ('x','y','z','intensity')
	colors = 'Spectral' 

	def __init__(self, 
		node_name = node_name, 
		topic = topic, 
		base_frame = base_frame,
		map_frame = map_frame,
		min_distance = min_distance,
		fields = fields, 
		colors = colors
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
		## params
		self.node_name = node_name
		self.topic = topic
		self.base_frame = base_frame
		self.map_frame = map_frame
		self.fields = fields
		self.colors = cm.get_cmap(colors)
		self.min_distance = min_distance
		
		# states
		self.cloud_msg = None
		self.new_msg = False
		self.pos = None
		self.rot = None
		
		## report
		self.report = {
			'received messages':0,
			'received points':0,
			'processed messages':0,
			'processed points':0,
			'rejected messages':0,
			'lost messages':0,
		}
		
		## properties
		self.worker = Thread(target=self.__job__)
		self.ready = Condition()
		self.listener = tf.TransformListener()
		self.pub = rospy.Publisher('{}/mesh'.format(self.node_name), TriangleMeshStamped, queue_size=10)
		rospy.Subscriber(self.topic, PointCloud2, self.__update__, queue_size=1)
		self.worker.start()

	def __update__(self, cloud_msg):
		'''
		Update routine, to be called by subscribers.
		
		Args:
			cloud [PointCloud2]: The ROS message 'sensor_msgs.msg.PointCloud2'.
		Publishes:
			TriangleMesh
		'''
		## report
		self.report['received messages'] += 1
		self.report['received points'] += cloud_msg.width
		
		## update pos
		try:
			pos, quat = self.listener.lookupTransform(self.map_frame, self.base_frame, rospy.Time(0))
		except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
			rospy.logwarn("Node '{}': {}".format(self.node_name, e))
			return
		
		if self.pos is not None and magnitude(self.pos - pos) < self.min_distance**2:
			self.report['rejected messages'] += 1
		elif self.ready.acquire(False):
			self.pos = np.array(pos)
			self.rot = R.from_quat(quat)
			self.cloud_msg = cloud_msg
			self.new_msg = True
			self.ready.notify()
			self.ready.release()
		else:
			rospy.logwarn("{} message dropped!".format(self.node_name))
			self.report['lost messages'] += 1
		pass
		
	def __job__(self):
		while not rospy.is_shutdown():
			self.ready.acquire()
			self.ready.wait(1.0)
			if self.new_msg:
				self.new_msg = False
			else:
				continue

			## report
			self.report['processed messages'] += 1
			self.report['processed points'] += cloud_msg.width
			
			## mesh
			X = numpify(self.cloud_msg)
			x, y, z, i = self.fields
			Q = np.array((X[x], X[y], X[z])).T
			P = sphere_uvd(Q)
					   
			surf = Delaunay(P[:,(0,1)])
			Ti = surf.simplices
			fN = face_normals(Q[Ti])
			vN = vec_normals(fN, Ti.flatten())
			
			## transform
			Q = self.rot.apply(Q)
			vN = self.rot.apply(vN)
			Q += self.pos

			## publish
			mesh = numpy_to_trianglemesh(Q, Ti, vN, P, self.colors(X[i]))
			mesh.header = self.cloud_msg.header
			mesh.header.frame_id = self.map_frame
			self.pub.publish(mesh)
		self.ready.release()
		pass
	
	def plot_report(self):
		report = ["\n+{:-^32}+".format(self.node_name)]
		report += ["|{:<20}{:>12}|".format(k, v) for k, v in self.report.iteritems()]
		report += ["+{:-^32}+\n".format("")]
		return "\n".join(report)

 
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
			default=MeshGen.node_name,
			help='Base name of the node.'
			)
		
		parser.add_argument(
			'--topic', '-t',
			metavar='TOPIC',
			default=MeshGen.topic,
			help='The topic to be subscribed.'
			)
		
		parser.add_argument(
			'--base_frame', '-b',
			metavar='STRING',
			default=MeshGen.base_frame,
			help='The base_frame where the point cloud cames from.'
			)
		
		parser.add_argument(
			'--map_frame', '-m',
			metavar='STRING',
			default=MeshGen.map_frame,
			help='The map_frame where the mesh is to plot at.'
			)
		
		parser.add_argument(
			'--min_distance', '-d',
			type=float,
			metavar='FLOAT',
			default=MeshGen.min_distance,
			help='Minimal travel distance to previous record.'
			)
		
		parser.add_argument(
			'--fields', '-f',
			metavar='STRING',
			nargs='+',
			default=MeshGen.fields,
			help='The field names of the PointCloud.'
			)
		
		parser.add_argument(
			'--colors', '-c',
			metavar='STRING',
			default=MeshGen.colors,
			help='The color mapping method.'
			)
		
		return parser


	args, _ = init_argparse().parse_known_args()
	rospy.init_node(args.node_name, anonymous=False)
	rospy.loginfo("Init node '{}' on topic '{}'".format(args.node_name, args.topic))
	node = MeshGen(**args.__dict__)
	rospy.loginfo("{} ready!".format(args.node_name))
	rospy.spin()
	rospy.loginfo(node.plot_report())
	rospy.loginfo("{} terminated!".format(args.node_name))
	
