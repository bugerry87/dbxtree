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
from time import sleep

#3rd-Party libs
import numpy as np
import pykitti  # install using pip install pykitti
from scipy.spatial import Delaunay
from sklearn import svm
from sklearn.cluster import DBSCAN

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
    N = np.cross(T[:,1] - T[:,0], T[:,2] - T[:,0])
    if normalize:
        return N / np.linalg.norm(N, axis=1)[:, None]
    else:
        return N


def main(args):
    # Load the data
    files = ifile(args.data, args.sort)    
    frames = pykitti.utils.yield_velo_scans(files)
    fig = viz.create_figure()
    db = DBSCAN(eps=0.3, n_jobs=10)
    plot = None

    for X in frames:
        if not len(X):
            break
        
        print("Generate Surface...")
        mesh = Delaunay(X[:,:2])
        Ti = mesh.simplices
        x = X[Ti,:3]
        N = face_normals(x)
        
        mp = viz.mesh(X, Ti, None, fig, None)
        input()
        
        print("Get Ground Plane...")
        x = x[:,0]
        mx = np.mean(x)
        vx = np.var(x)
        x = (x - mx) / vx
        SVM = svm.LinearSVR() #SVC(gamma='scale', kernel='poly', degree=3)
        print("    Fit SVM...")
        SVM.fit(x, N[:,2]<0.9, -x[:,2])
        print("    Classifiy...")
        G = SVM.predict(x) > 0
        Ti = Ti[G]
        
        mp = viz.mesh(X, Ti, None, fig, mp)
        input()
        
        mask = np.ones_like(X[:,0], bool)
        mask[Ti] = False
        x = X[mask]
        
        print("Clustering...")
        Y = db.fit(x[:,:3]).labels_
        
        print("Rendering...")
        viz.clear_figure(fig)
        plot = viz.vertices(x, Y, fig, plot)
        inp = myinput(
            "Press to continue: ",
            default=1,
            cast=int
            )
        
        if inp <= 0:
            break
        viz.clear_figure(fig)
    return 0

if __name__ == '__main__':
    import sys
    parser = init_argparse()
    args, _ = parser.parse_known_args()
    if len(sys.argv) == 1:
        args = validate(args)
    exit(main(args))
