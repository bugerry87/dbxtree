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
        frame_id='base_link', 
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
        self.worker = Thread(target=self.__job__)
        self.ready = Condition()
        self.new_msg = False
        
        ## init the node
        self.pub = rospy.Publisher('{}/mesh'.format(self.name), TriangleMeshStamped, queue_size=5)
        rospy.Subscriber(self.topic, PointCloud2, self.__update__, queue_size=5)
        self.worker.start()

    def __update__(self, cloud_msg):
        '''
        Update routine, to be called by subscribers.
        
        Args:
            cloud [PointCloud2]: The ROS message 'sensor_msgs.msg.PointCloud2'.
        Publishes:
            TriangleMesh
        '''
        if self.ready.acquire(False):
            self.cloud_msg = cloud_msg
            self.new_msg = True
            self.ready.notify()
            self.ready.release()
        
    def __job__(self):
        while not rospy.is_shutdown():
            self.ready.acquire()
            self.ready.wait(1.0)
            if self.new_msg:
                self.new_msg = False
            else:
                continue

            next(self.delta)
            X = numpify(self.cloud_msg)
            x, y, z, _ = self.fields
            Q = np.array((X[x], X[y], X[z])).T
            print('numpify:', next(self.delta), 'shape:', Q.shape)
            
            P = spatial.sphere_uvd(Q)
            Y = spatial.prob(P[:,2])
            P[:,:2] *= P.max() / np.pi
            surf = Delaunay(P[:,(0,1)])
            print('Delaunay:', next(self.delta))
            Ti = surf.simplices
            fN = spatial.face_normals(Q[Ti])
            vN = spatial.vec_normals(fN, Ti.flatten())
            print('normals:', next(self.delta))
            Mask = spatial.mask_planar(vN, fN, Ti.flatten(), 0.95)
            P = P[Mask]
            Q = Q[Mask]
            Y = Y[Mask]
            print('surf plannar:', next(self.delta), 'shape:', Q.shape)
            
            surf = Delaunay(P[:,(0,1)])
            Ti = surf.simplices
            fN = spatial.face_normals(Q[Ti])
            vN = spatial.vec_normals(fN, Ti.flatten())
            print('meshed:', next(self.delta))

            mesh = numpy_to_trianglemesh(Q, Ti, vN, P, self.cmap(Y))
            mesh.header = self.cloud_msg.header
            print('seq', mesh.header.seq)
            mesh.header.frame_id = self.frame_id
            print('message:', next(self.delta))
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
            default='base_link',
            help='The frame_id for rviz to plot the markers at.'
            )
        
        return parser


    args, _ = init_argparse().parse_known_args()
    rospy.init_node(args.base_name, anonymous=False)
    rospy.loginfo("Init node '{}' on topic '{}'".format(args.base_name, args.topic))
    mesh_gen = MeshGen(args.base_name, args.topic, args.frame_id)
    rospy.loginfo("Node '{}' ready!".format(args.base_name))
    rospy.spin()
