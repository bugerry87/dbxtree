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

#Local libs
import cluster
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


def limit_split(X, depth=4):
    C = []
    def recursive(x, depth):
        c = cluster.init_kmeans(x, 0, 'limits')
        for Y, c, delta, step in cluster.kmeans(x, c, 0, 1):
            pass
        C.append(c)
        if depth:
            for i in np.unique(Y):
                recursive(x[Y==i,:], depth-1)
    
    recursive(X, depth)
    C = np.stack(C)
    C = C.reshape((C.shape[0] * C.shape[1], C.shape[2]))
    return C


def main(args):
    # Load the data
    files = ifile(args.data, args.sort)    
    frames = pykitti.utils.yield_velo_scans(files)
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))

    frame = next(frames, [])
    if not len(frame):
        print("Error: Empty frame!")
        return 1

    plot = mlab.points3d(
        frame[:,0],
        frame[:,1],
        frame[:,2],
        frame[:,3],
        mode="point",         # How to render each point {'point', 'sphere' , 'cube' }
        colormap='spectral',  # 'bone', 'copper', 'spectral'
        scale_factor=100,     # scale of the points
        line_width=10,        # Scale of the line, if any
        figure=fig,
        )
    
    @mlab.animate(delay=10)
    def animation():
        _frame = frame[:,:3]
        C = limit_split(_frame[:,:2], 1)
        
        while len(_frame):
            for Y, C, delta, step in cluster.kmeans(_frame[:,:2], C, 0, 3):
                Y[0] = -1
                plot.mlab_source.reset(
                    x = _frame[:, 0],
                    y = _frame[:, 1],
                    z = _frame[:, 2],
                    scalars = Y
                )
                fig.render()
                yield
            _frame = next(frames, [])
        
        print("Done")
    
    for frame in frames:
        break
        if not len(frame):
            break
        
        animator = animation()
        next_frame = myinput(
            "Enter for next frame or 0 to quit: ",
            default=1,
            cast=int
            )
        if not next_frame:
            break
    
    next_frame = myinput(
            "Start Animation:",
            default=1,
            cast=int
            )
    if next_frame:         
        animator = animation()
        myinput("Press any key to quit:")
    return 0

if __name__ == '__main__':
    import sys
    parser = init_argparse()
    args, _ = parser.parse_known_args()
    if len(sys.argv) == 1:
        args = validate(args)
    exit(main(args))
