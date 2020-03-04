#!/usr/bin/env python
"""
Simulate event driven LiDAR

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


def polarize(X, scale=(10,10)):
    P = X.copy()
    P[:,0] = np.arccos(X[:,0] / np.linalg.norm(X[:,(0,1)], axis=1)) * (1*(X[:,1] >= 0) - (X[:,1] < 0)) * scale[0]
    P[:,1] = np.linalg.norm(X[:,:2], axis=1)
    P[:,2] = np.arcsin(P[:,2] / P[:,1]) * scale[1]
    return P


def quantirize(P, t):
    k = P[0]
    Q = [k]
    
    for p0, p1 in zip(P[1:-1], P[2:]):
        pp = p1 - p0
        pk = p0 - k
        pp = pp / np.linalg.norm(pp)
        pk = pk / np.linalg.norm(pk)
        np.dot(pp, pk)
        
        if pk[0] < 0:
            #new line
            Q.append(p)
        elif pk[1] > g + b and pk[1] < g - b:
            #new keypoint found
            k = p
            a = pk[0] / pk[1]
            Q.append(k)
        else:
            continue
    return np.array(Q)


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
        P = polarize(X)
        P[:, 1] = 0
        Y = np.arange(P.shape[0])
        
        print("Plot polar...")
        viz.vertices(X, Y, fig, None)
        if input():
            break
        viz.clear_figure(fig)
        
        break
    return 0

if __name__ == '__main__':
    import sys
    parser = init_argparse()
    args, _ = parser.parse_known_args()
    if len(sys.argv) == 1:
        args = validate(args)
    exit(main(args))
