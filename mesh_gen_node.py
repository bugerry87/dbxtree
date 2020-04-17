#!/usr/bin/env python3

#build in
import sys

#installed
import numpy as np
from scipy.spatial import Delaunay
from matplotlib import cm

#ros
import rospy
from std_msgs.msg import ColorRGBA
from mesh_msgs.msg import TriangleMeshStamped, TriangleIndices
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point
from ros_numpy import numpify
from utils import *

#local
import spatial


def numpy_to_trianglemesh(verts, trids, norms, uvds=None, colors=None):
    mesh = TriangleMeshStamped()
    mesh.mesh.vertices = [Point(*p) for p in verts]
    mesh.mesh.vertex_normals = [Point(*n) for n in norms]
    mesh.mesh.triangles = [TriangleIndices(t) for t in trids.tolist()]
    if not uvds is None:
        mesh.mesh.vertex_texture_coords = [Point(*uvd) for uvd in uvds]
    if not colors is None:
        mesh.mesh.vertex_colors = [ColorRGBA(*colors) for colors in colors]
    return mesh


class MeshGen:
    def __init__(self, 
        name='MeshGen', 
        topic='~/velodyne_points', 
        frame_id='map', 
        fields=('x','y','z','intensity'), 
        cmap='Spectral'
        ):
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
        self.fields = fields
        self.cmap = cm.get_cmap(cmap)
        self.delta = time_delta(time())
        
        ## init the node
        self.pub = rospy.Publisher('{}/mesh'.format(self.name), TriangleMeshStamped, queue_size=10)
        rospy.Subscriber(self.topic, PointCloud2, self.__update__)

    def __update__(self, cloud_msg):
        '''
        Update routine, to be called by subscribers.
        
        Args:
            cloud [PointCloud2]: The ROS message 'sensor_msgs.msg.PointCloud2'.
        Publishes:
            TriangleMesh
        '''
        
        next(self.delta)
        X = numpify(cloud_msg)
        x, y, z, i = self.fields
        Q = np.array((X[x], X[y], X[z])).T
        Y = spatial.prob(X[i])
        print('numpify:', next(self.delta), 'shape:', Q.shape)
        
        P = spatial.sphere_uvd(Q)
        surf = Delaunay(P[:,(0,1)])
        print('Delaunay:', next(self.delta))
        Ti = surf.simplices
        fN = spatial.face_normals(Q[Ti])
        eN = spatial.edge_normals(fN, Ti.flatten())
        print('normals:', next(self.delta))
        Mask = spatial.mask_planar(eN, fN, Ti.flatten(), 0.95)
        P = P[Mask]
        Q = Q[Mask]
        Y = Y[Mask]
        print('surf plannar:', next(self.delta), 'shape:', Q.shape)
        
        surf = Delaunay(P[:,(0,1)])
        Ti = surf.simplices
        fN = spatial.face_normals(Q[Ti])
        eN = spatial.edge_normals(fN, Ti.flatten())
        print('meshed:', next(self.delta))

        mesh = numpy_to_trianglemesh(P, Ti, eN, P, self.cmap(Y))
        mesh.header = cloud_msg.header
        mesh.header.frame_id = self.frame_id
        print('message:', next(self.delta))
        self.pub.publish(mesh)
        input()
        
 
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
    rospy.spin()
