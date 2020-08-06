#!/usr/bin/env python2


## build in
from threading import Thread, Condition

## installed
import numpy as np

## ros
import rospy
import tf
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point, Quaternion


def magnitude(X, sqrt=False):
	if len(X.shape) == 1:
		m = np.sum(X**2)
	else:
		m = np.sum(X**2, axis=-1).reshape(*(X.shape[:-1] + (1,)))
	return np.sqrt(m) if sqrt else m


class MapPCD:
	## defaults
	node_name = 'MapPCD'
	topic = '~/velodyne_points'
	base_frame = 'base_link'
	map_frame = 'map'
	min_distance = 0.125

	def __init__(self, 
		node_name = node_name, 
		topic = topic, 
		base_frame = base_frame,
		map_frame = map_frame,
		min_distance = min_distance
		):
		'''
		Initialize an MapPCD node.
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
		self.min_distance = min_distance
		
		# states
		self.cloud_msg = None
		self.new_msg = False
		self.pos = None
		self.rot = None
		self.seq = 0
		
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
		self.pub_pcd = rospy.Publisher('{}/pcd'.format(self.node_name), PointCloud2, queue_size=10)
		self.pub_odom = rospy.Publisher('{}/odom'.format(self.node_name), Odometry, queue_size=10)
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
			self.quat = quat
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
			self.seq += 1
			self.report['processed messages'] = self.seq
			self.report['processed points'] += self.cloud_msg.width
			
			## cloud
			self.cloud_msg.header.frame_id = self.base_frame
			self.cloud_msg.header.seq = self.seq
			
			## odom
			odom = Odometry()
			odom.header.frame_id = self.map_frame
			odom.header.stamp = self.cloud_msg.header.stamp
			odom.header.seq = self.seq
			odom.pose.pose.position = Point(*self.pos)
			odom.pose.pose.orientation = Quaternion(*self.quat)

			## publish
			self.pub_pcd.publish(self.cloud_msg)
			self.pub_odom.publish(odom)
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
			default=MapPCD.node_name,
			help='Base name of the node.'
			)
		
		parser.add_argument(
			'--topic', '-t',
			metavar='TOPIC',
			default=MapPCD.topic,
			help='The topic to be subscribed.'
			)
		
		parser.add_argument(
			'--base_frame', '-b',
			metavar='STRING',
			default=MapPCD.base_frame,
			help='The base_frame where the point cloud cames from.'
			)
		
		parser.add_argument(
			'--map_frame', '-m',
			metavar='STRING',
			default=MapPCD.map_frame,
			help='The map_frame where the mesh is to plot at.'
			)
		
		parser.add_argument(
			'--min_distance', '-d',
			type=float,
			metavar='FLOAT',
			default=MapPCD.min_distance,
			help='Minimal travel distance to previous record.'
			)
		
		return parser


	args, _ = init_argparse().parse_known_args()
	rospy.init_node(args.node_name, anonymous=False)
	rospy.loginfo("Init node '{}' on topic '{}'".format(args.node_name, args.topic))
	node = MapPCD(**args.__dict__)
	rospy.loginfo("{} ready!".format(args.node_name))
	rospy.spin()
	rospy.loginfo(node.plot_report())
	rospy.loginfo("{} terminated!".format(args.node_name))
	
