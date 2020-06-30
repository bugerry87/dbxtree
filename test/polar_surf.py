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
from scipy.spatial import Delaunay

#Local libs
import viz
import spatial
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
        #description="Demo for embedding data via LDA",
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


def main(args):
    # Load the data
    files = ifile(args.data, args.sort)    
    frames = pykitti.utils.yield_velo_scans(files)
    fig = viz.create_figure()
    plot = None

    for X in frames:
        if not len(X):
            break
        
        print("Input size:", X.shape)
        print("Plot X...")
        viz.vertices(X, X[:,3], fig, None)
        if input():
            break
        viz.clear_figure(fig)
        
        P = spatial.sphere_uvd(X[:,(1,0,2)])
        P[:,(0,1)] *= P.max() / np.pi
        
        print("Plot polar...")
        viz.vertices(P, X[:,3], fig, None)
        if input():
            break
        viz.clear_figure(fig)
        
        print("Generate Surface...")
        mesh = Delaunay(P[:,(0,1)])
        Ti = mesh.simplices
        
        viz.mesh(P, Ti, None, fig, None)
        if input():
            break
        viz.clear_figure(fig)
        
        print("Remove planars...")
        fN = spatial.face_normals(X[Ti,:3])
        eN = spatial.edge_normals(fN, Ti.flatten())
        Mask = spatial.mask_planar(eN, fN, Ti.flatten(), 0.90)
        P = P[Mask]
        X = X[Mask]
        mesh = Delaunay(P[:,(0,1)])
        Ti = mesh.simplices
        
        print("New size:", X.shape)
        viz.mesh(X, Ti, None, fig, None)
        if input():
            break
        viz.clear_figure(fig)
        
        print("Remove Narrow Z-faces...")
        fN = spatial.face_normals(P[Ti,:3])
        Mask = np.abs(fN[:,1]) > 0.2
        Ti = Ti[Mask]
        
        viz.mesh(P[:,(2,0,1)], Ti, None, fig, None)
        if input():
            break
        viz.clear_figure(fig)
        
        print("Unfold View...")
        viz.mesh(X, Ti, None, fig, None)
        if input():
            break
        viz.clear_figure(fig)
        
        print("Final shape:", (np.unique(Ti.flatten())).shape)
        
        break
    return 0

if __name__ == '__main__':
    import sys
    parser = init_argparse()
    args, _ = parser.parse_known_args()
    if len(sys.argv) == 1:
        args = validate(args)
    exit(main(args))
