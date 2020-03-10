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

#Local libs
import viz
from utils import *
from mesh import *


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


def quantirize(P, m=1):
    k = P[0]
    p0 = P[1]
    Q = [k]
    p0k, mag = norm(p0 - k, True)
    
    for p1 in P[2:]:
        pp, ppm = norm(p1 - p0, True)
        mag += ppm
        
        p1k = norm(p1 - k)
        dot = np.dot(p0k, p1k)
        
        if dot < 1 - np.exp(-mag/m):
            #new keypoint detected
            k = p0
            p0 = p1
            p0k = pp
            mag = ppm
            Q.append(k)
        else:
            #update
            p0 = p1
            p0k = p1k
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
        
        m = 1
        while m > 0:
            print("Input size:", X.shape)
            print("Org plot...")
            Y = np.arange(X.shape[0])
            plot = viz.lines(X, Y, fig, None)
            
            m = myinput(
                    "Set the magnitude: ",
                    cast=float,
                    default=False
                )
            
            if m <= 0:
                break
        
            print("Magnitude:", m)
            Q = quantirize(X, m)
            Y = np.arange(Q.shape[0])
            Qi = np.array((range(Q.shape[0]-1), range(1,Q.shape[0]))).T
            
            viz.lines(Q, Y, fig, plot)
            print("Output size:", Q.shape)
            
            print("Compute loss...")
            L, mp, nn = nn_point2line(Q, Qi, X)
            print("Loss mean:", L.mean())
            
            if input():
                break
            viz.clear_figure(fig)
            
            print("Plot polar...")
            P = polarize(Q)
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
            fN = face_normals(Q[Ti,:3])
            eN = edge_normals(fN, Ti.flatten())
            Mask = mask_planar(eN, fN, Ti.flatten(), 0.9)
            P = P[Mask]
            Q = Q[Mask]
            mesh = Delaunay(P[:,(0,2)])
            Ti = mesh.simplices
            viz.mesh(Q, Ti, None, fig, None)
            print("New size:", Q.shape)
            if input():
                break
            viz.clear_figure(fig)
            
            print("Remove Narrow Z-faces...")
            fN = face_normals(P[Ti,:3])
            Mask = np.abs(fN[:,1]) > 0.1
            Ti = Ti[Mask]
            viz.mesh(P[:,(2,0,1)], Ti, None, fig, None)
            print("Final size:", np.unique(Ti.flatten()).shape)
            if input():
                break
            viz.clear_figure(fig)
            
            print("Unfold View...")
            viz.mesh(Q, Ti, None, fig, None)
            
            if input():
                break
            viz.clear_figure(fig)
            
        viz.clear_figure(fig)  
        if m < 0:
            break
    return 0

if __name__ == '__main__':
    import sys
    parser = init_argparse()
    args, _ = parser.parse_known_args()
    if len(sys.argv) == 1:
        args = validate(args)
    exit(main(args))
