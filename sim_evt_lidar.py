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


def quantirize(P, t=0.9, m=0):
    k = P[0]
    p0 = P[1]
    Q = [k]
    
    for p1 in P[2:]:
        pp = p1 - p0
        ppm = np.linalg.norm(pp)
        if ppm <= m:
            continue
        
        p0 = p1
        pk = p0 - k
        pp = pp / ppm
        pk = pk / np.linalg.norm(pk)
        dot = np.dot(pp, pk)
        
        if dot >= t:
            #reject point
            p0 = p1
            continue
        
        #else new line
        k = p0
        p0 = p1
        Q.append(k)
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
        
        print("Org plot...")
        Y = np.arange(X.shape[0])
        plot = viz.lines(X, Y, fig, None)
        if input():
            break
        #viz.clear_figure(fig)
        
        print("Plot quantirized...")
        Q = quantirize(X, 0.9, 0.05)
        Y = np.arange(Q.shape[0])
        
        viz.lines(Q, Y, fig, plot)
        print("Output size:", Q.shape)
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
