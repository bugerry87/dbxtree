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
from threading import Lock

#3rd-Party libs
import numpy as np
import pykitti  # install using pip install pykitti
from scipy.spatial import Delaunay

#Local libs
import viz
from utils import *
from spatial import *
from KDNTree import KDNTree


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


def quantirize_old(P, m=1):
    k = P[0]
    p0 = P[1]
    p0k, mag = norm(p0 - k, True)
    mask = np.zeros(P.shape[0], dtype=bool)
    
    for i, p1 in enumerate(P[2:]):
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
            mask[i] = True
        else:
            #update
            p0 = p1
            p0k = p1k
    return P[mask]

def quantirize(P, m=1):
    k = P[0]
    p0 = P[1]
    p0k = p0 - k
    p0km = magnitude(p0k)
    mag = p0km
    mask = np.zeros(P.shape[0], dtype=bool)
    m = m**2
    
    for i, p1 in enumerate(P[2:]):
        pp = p1 - p0
        ppm = magnitude(pp)
        mag += ppm
        
        p1k = p1 - k
        p1km = magnitude(p1k)
        dot = np.dot(p0k, p1k)**2 / (p0km * p1km) 
        
        if dot < 1 - np.exp(-mag/m)**4:
            #new keypoint detected
            k = p0
            p0 = p1
            p0k = pp
            p0km = ppm
            mag = ppm
            mask[i] = True
        else:
            #update
            p0 = p1
            p0k = p1k
            p0km = p1km
    return P[mask]


def main(args):
    # Load the data
    files = ifile(args.data, args.sort)    
    frames = pykitti.utils.yield_velo_scans(files)
    fig = viz.create_figure()
    plot = None
    
    print_lock = Lock()
    main.last = 0
    def callback(tree):
        print_lock.acquire()
        curr = int(tree.done.mean() * 50)
        dif = curr - main.last
        if curr > main.last:
            print('#' * dif, end='', flush=True)
        main.last = curr
        print_lock.release()

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
            Q = quantirize(X[:,:3], m)
            Y = np.arange(Q.shape[0])
            Qi = np.array((range(Q.shape[0]-1), range(1,Q.shape[0]))).T
            
            viz.lines(Q, Y, fig, plot)
            print("Output size:", Q.shape)
            
            #print("Compute loss...")
            #L, mp, nn = nn_point2line(Q, Qi, X)
            tree = KDNTree(Q, Qi, j=8, leaf_size=100)
            #print("\n0%                      |50%                     |100%")
            #L, mp, nn = tree.query(X[:,:3], callback=callback)
            
            #print("\nLoss mean:", L.mean())
            
            if input():
                break
            viz.clear_figure(fig)
            
            print("Plot polar...")
            P = polarize(Q)
            viz.vertices(P[:,(1,0,2)], P[:,2], fig, None)
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
            fN = face_normals(Q[Ti], True)
            eN = edge_normals(fN, Ti.flatten(), True)
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
            fN = face_normals(P[Ti], True)
            #Mask = (fN[:,2] > -0.8) #& (np.abs(fN[:,1]) > 0.05)
            #Ti = Ti[Mask]
            viz.mesh(P, Ti, None, fig, None)
            print("Final size:", np.unique(Ti.flatten()).shape)
            if input():
                break
            viz.clear_figure(fig)
            
            print("Unfold View...")
            fN = face_normals(Q[Ti], True)
            eN = edge_normals(fN, Ti.flatten(), True)
            viz.mesh(Q, Ti, eN[:,2], fig, None)
            
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
