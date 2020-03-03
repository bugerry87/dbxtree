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


def face_normals(T, normalize=True):
    fN = np.cross(T[:,1] - T[:,0], T[:,2] - T[:,0])
    if normalize:
        return fN / np.linalg.norm(fN, axis=1)[:, None]
    else:
        return fN


def edge_normals(fN, Ti_flat, normalize=True):
    fN = fN.repeat(3, axis=0)
    eN = np.zeros((Ti_flat.max()+1, 3))
    for fn, i in zip(fN, Ti_flat):
        eN[i] += fn
    if normalize:
        return eN / np.linalg.norm(eN, axis=1)[:, None]
    else:
        return eN


def mask_planar(eN, fN, Ti_flat, min_dot=0.9, mask=None):
    fN = fN.repeat(3, axis=0)
    if mask is None:
        mask = np.ones(Ti_flat.max()+1, dtype=bool)
    for fn, i in zip(fN, Ti_flat):
        if mask[i]:
            mask[i] &= np.dot(eN[i], fn) <= min_dot
        else:
            pass
    return mask


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
        
        P = X.copy()
        P[:,0] = np.arccos(X[:,0] / np.linalg.norm(X[:,(0,1)], axis=1)) * (1*(X[:,1] >= 0) - (X[:,1] < 0))
        P[:,1] = np.linalg.norm(X[:,:2], axis=1)
        P[:,2] = np.arcsin(P[:,2] / P[:,1]) * 10
        P[:,3] = 1*(X[:,1] >= 0) + 2*(X[:,1] < 0) + 4*(X[:,2] >= 0) + 8*(X[:,2] < 0)
        P[0,3] = 0
        P[:,0] *= 10
        
        print("Plot polar...")
        viz.vertices(P[:,(1,0,2)], P[:,3], fig, None)
        if input():
            break
        viz.clear_figure(fig)
        
        print("Generate Surface...")
        mesh = Delaunay(P[:,(0,2)])
        Ti = mesh.simplices
        
        viz.mesh(P[:,(2,0,1)], Ti, None, fig, None)
        if input():
            break
        viz.clear_figure(fig)
        
        print("Remove planars...")
        fN = face_normals(X[Ti,:3])
        eN = edge_normals(fN, Ti.flatten())
        Mask = mask_planar(eN, fN, Ti.flatten(), 0.9)
        P = P[Mask]
        X = X[Mask]
        mesh = Delaunay(P[:,(0,2)])
        Ti = mesh.simplices
        
        viz.mesh(X, Ti, None, fig, None)
        if input():
            break
        viz.clear_figure(fig)
        
        print("Remove Narrow Z-faces...")
        fN = face_normals(P[Ti,:3])
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
