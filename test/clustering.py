#!/usr/bin/env python
"""
VISUALISE THE LIDAR DATA FROM THE ANY DATASET

Based on the sample code from
    https://github.com/utiasSTARS/pykitti/blob/master/demos/demo_raw.py
And:
    http://stackoverflow.com/a/37863912
    
Author: Gerald Baulig
"""

#Standard libs
from argparse import ArgumentParser

#3rd-Party libs
import numpy as np
import pykitti  # install using pip install pykitti
from mayavi import mlab
from scipy.spatial import Delaunay
from sklearn.cluster import DBSCAN as CLUSTER

#Local libs
from utils import *


def init_argparse(parents=[]):
    ''' init_argparse(parents=[]) -> parser
    Initialize an ArgumentParser for this module.
    
    Args:
        parents: A list of ArgumentParsers of other scripts, if there are any.
        
    Returns:
        parser: The ArgumentParsers.
    '''
    parser = ArgumentParser(
        description="Demo for embedding data via LDA",
        parents=parents
        )
    
    parser.add_argument(
        '--data', '-X',
        metavar='WILDCARD',
        help="Wildcard to the LiDAR files.",
        default='**.bin'
        )
    
    parser.add_argument(
        '--sort', '-s',
        metavar='BOOL',
        nargs='?',
        type=bool,
        help="Sort the files?",
        default=False,
        const=True
        )
    
    return parser


def validate(args):
    args.data = myinput(
        "Wildcard to the LiDAR files.\n" + 
        "    data ('**.bin'): ",
        default='**.bin'
        )
    
    args.sort = myinput(
        "Sort the files?\n" + 
        "    sort (False): ",
        default=False
        )
    return args


def plot_vertices(X, Y, fig, plot=None):
    if not len(X):
        raise ValueError("Error: Empty frame!")

    if plot == None:
        plot = mlab.points3d(
            X[:,0],
            X[:,1],
            X[:,2],
            Y,
            mode="point",         # How to render each point {'point', 'sphere' , 'cube' }
            colormap='spectral',  # 'bone', 'copper',
            scale_factor=100,     # scale of the points
            line_width=10,        # Scale of the line, if any
            figure=fig,
            )
    else:
        plot.mlab_source.reset(
            x = X[:, 0],
            y = X[:, 1],
            z = X[:, 2],
            scalars = Y
        )
    fig.render()
    return plot


def plot_mesh(X, T, Y, fig, plot=None):
    if not len(X):
        raise ValueError("Error: Empty frame!")

    if plot == None:
        plot = mlab.triangular_mesh(
            X[:,0],
            X[:,1],
            X[:,2],
            T,
            #color=Y,
            colormap='spectral',  # 'bone', 'copper',
            line_width=10,        # Scale of the line, if any
            figure=fig,
            )
    else:
        plot.mlab_source.reset(
            x = X[:, 0],
            y = X[:, 1],
            z = X[:, 2],
            scalars = Y
        )
    fig.render()
    return plot


def main(args):
    # Load the data
    files = ifile(args.data, args.sort)    
    frames = pykitti.utils.yield_velo_scans(files)
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
    clu = CLUSTER(eps=0.3, n_jobs=10)
    plot = None

    for X in frames:
        if not len(X):
            break
            
        #X = X[X[:,2] > np.mean(X[:,2])]
        mesh = Delaunay(X[:,:2])
        T = mesh.simplices
        N = mesh.equations
        print(N)
        #print("Clustering...")
        #Y = clu.fit(X[:,:3]).labels_
        print("Rendering...")
        plot = plot_mesh(X, T, Y, fig, plot)
        inp = myinput(
            "Press to continue: ",
            default=1,
            cast=int
            )
        
        if inp <= 0:
            break

    return 0

if __name__ == '__main__':
    import sys
    parser = init_argparse()
    args, _ = parser.parse_known_args()
    if len(sys.argv) == 1:
        args = validate(args)
    exit(main(args))
