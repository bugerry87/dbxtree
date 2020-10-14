#!/usr/bin/env python3

#installed
import numpy as np

#ros
import rospy
#from ros_numpy.point_cloud2 import pointcloud2_to_array, array_to_pointcloud2
from sensor_msgs.msg import PointCloud2


def pointcloud2_to_numpy(cloud_msg, min_dist=0.0):
    '''
    Converts a rospy PointCloud2 message to a numpy array
    Reshapes the returned array to have shape (height, width), even if the height is 1.
    The reason for using np.fromstring rather than struct.unpack is speed... especially
    for large point clouds, this will be <much> faster.
    
    Args:
        cloud_msg [PointCloud2]
    Returns:
        np.ndarray
    '''
    
    cloud_arr = np.fromstring(cloud_msg.data, np.float32, cloud_msg.width*4)
    cloud_arr = cloud_arr.reshape(cloud_msg.width, -1)
    return cloud_arr


class ReMapper:
    def __init__(self, name='ReMapper', topic='~/cloud_pcd', frame_id='map', fields=(0,1,2)):
        self.name = name
        self.topic = topic
        self.frame_id = frame_id
        self.fields = fields
        
        ## init the node
        self.pub = rospy.Publisher('{}/pcl2'.format(self.name), PointCloud2, queue_size=10)
        rospy.Subscriber(self.topic, PointCloud2, self.__swap_axis__)

    def __swap_axis__(self, msg):
        cloud_arr = pointcloud2_to_numpy(msg)
        cloud_arr = cloud_arr[:,self.fields]
        
        msg.header.frame_id = self.frame_id if self.frame_id else msg.header.frame_id
        msg.fields = [msg.fields[field] for field in self.fields]
        msg.data = cloud_arr.tostring()
        
        print("PC2 {} remapped: to '{}'".format(msg.header.stamp, msg.fields))
        self.pub.publish(msg)
        
 
if __name__ == '__main__':
    from argparse import ArgumentParser
    import matplotlib.pyplot as plt
    
    def init_argparse(parents=[]):
        ''' init_argparse(parents=[]) -> parser
        Initialize an ArgumentParser for this module.
        
        Args:
            parents: A list of ArgumentParsers of other scripts, if there are any.
            
        Returns:
            parser: The ArgumentParsers.
        '''
        parser = ArgumentParser(
            description="ROS node to swap the axis of a PointCloud2.",
            parents=parents
            )
        
        parser.add_argument(
            '--base_name', '-n',
            metavar='STRING',
            default='ReMapper',
            help='Base name of the node.'
            )
        
        parser.add_argument(
            '--topic', '-t',
            metavar='TOPIC',
            default='~/cloud_pcd',
            help='The topic to be subscribed.'
            )
        
        parser.add_argument(
            '--frame_id', '-f',
            metavar='STRING',
            default=None,
            help='The frame_id for rviz to plot the markers at.'
            )
        
        parser.add_argument(
            '--fields', '-a',
            metavar='INT',
            type=int,
            nargs='*',
            default=(0,1,2),
            help='The re-arangement of the fields by name.'
            )
        
        return parser


    args, _ = init_argparse().parse_known_args()
    rospy.init_node(args.base_name, anonymous=False)
    rospy.loginfo("Init node '{}' on topic '{}'".format(args.base_name, args.topic))
    ReMapper(args.base_name, args.topic, args.frame_id, args.fields)
    rospy.loginfo("Node '{}' ready!".format(args.base_name))
    rospy.spin()
