#!/usr/bin/env python3

#build in
import sys

#installed
import numpy as np
from scipy.spatial import Delaunay

#ros
import rospy
from std_msgs.msg import ColorRGBA
from mesh_msgs.msg import TriangleMesh
from sensor_msgs.msg import PointCloud2

#local
import spatial


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
    return cloud_arr[np.sum(cloud_arr[:,:3]**2, axis=1) > min_dist**2]


def numpy_to_trianglemesh(verts, trids, norms, uvds=None):
    mesh = TriangleMesh()
    mesh.vertices = (Point(*p) for p in verts)
    mesh.vertex_normals = (Point(*n) for n in norms)
    mesh.triangles = (TriangleIndices(*t) for t in trids)
    if not uvs is None:
        mesh.vertex_texture_coords = (Point(*uvd) for uvd in uvds)
    return mesh


class MeshGen:
    def __init__(self, name='MeshGen', topic='~/velodyne_points', frame_id='map'):
        '''
        Initialize an MeshGen node.
        Subscribes velodyne_point cloud data.
        Publishes mesh_tool.msgs.
        
        Args:
            name [str]:     Base name of the node.
            topic [str]:    The topic to be subscribed.
            frame_id [str]: The frame_id in rviz where the markers get plotted at.
        '''
        self.name = name
        self.topic = topic
        self.frame_id = frame_id
        self.reset()
        
        ## init the node
        self.pub = rospy.Publisher('{}/mesh'.format(self.name), TriangleMesh, queue_size=100)
        rospy.Subscriber(self.topic, PointCloud2, self.__update__)
    
    def reset(self):
        '''
        Resets the assumed IMU state.
        '''
        pass

    def __update__(self, cloud_msg):
        '''
        Update routine, to be called by subscribers.
        
        Args:
            cloud [PointCloud2]: The ROS message 'sensor_msgs.msg.PointCloud2'.
        Publishes:
            TriangleMesh
        '''
        
        Q = pointcloud2_to_numpy(cloud_msg)
        Q, Y = Q[:,:3], Q[:,3]
        print(np.any(np.isnan(Q)))
        P = spatial.sphere_uvd(Q, True)
        print(np.any(np.isnan(P)))
        
        surf = Delaunay(P[:,(0,1)])
        Ti = surf.simplices
        fN = spatial.face_normals(Q[Ti], True)
        eN = spatial.edge_normals(fN, Ti.flatten(), True)
        Mask = spatial.mask_planar(eN, fN, Ti.flatten(), 0.95)
        
        P = P[Mask]
        Q = Q[Mask]
        surf = Delaunay(P[:,(0,1)])
        Ti = surf.simplices
        fN = spatial.face_normals(Q[Ti])
        eN = spatial.edge_normals(fN, Ti.flatten())

        mesh = numpy_to_trianglemesh(Q, Ti, eN, uvd)
        self.pub.publish(mesh)
        
 
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
            description="ROS node to transform PointCloud2 to TriangleMash.",
            parents=parents
            )
        
        parser.add_argument(
            '--base_name', '-n',
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
            default='map',
            help='The frame_id for rviz to plot the markers at.'
            )
        
        return parser


    args, _ = init_argparse().parse_known_args()
    rospy.init_node(args.base_name, anonymous=False)
    rospy.loginfo("Init node '{}' on topic '{}'".format(args.base_name, args.topic))
    mesh_gen = MeshGen(args.base_name, args.topic, args.frame_id)
    rospy.loginfo("Node '{}' ready!".format(args.base_name))
    
    while True:
        inp = input("\n".join((
            "Press 'r' to reset MeshGen state,",
            "  or 'q' for quit (q): ")))
        if inp == 'r':
            mesh_gen.reset()
            rospy.loginfo("Reset node: '{}'!".format(args.base_name))
        else:
            break
