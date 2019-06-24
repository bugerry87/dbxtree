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
        colormap='spectral',  # 'bone', 'copper',
        scale_factor=100,     # scale of the points
        line_width=10,        # Scale of the line, if any
        figure=fig,
        )
    
    next_frame = 1
    for frame in frames:
        if not len(frame):
            break
        elif next_frame:
            next_frame = myinput(
                "Enter to continue or 0 for animation: ",
                default=1,
                cast=int
                )
        else:
            break
        
        plot.mlab_source.reset(
            x = frame[:, 0],
            y = frame[:, 1],
            z = frame[:, 2],
            scalars = frame[:, 3]
        )
        fig.render() 

    @mlab.animate(delay=10)
    def animation():
        _frame = frame
        while len(_frame):
            plot.mlab_source.reset(
                x = _frame[:, 0],
                y = _frame[:, 1],
                z = _frame[:, 2],
                scalars = _frame[:, 3]
            )
            fig.render()
            yield
            _frame = next(frames, [])

    animator = animation()
    mlab.show()
    return 0

if __name__ == '__main__':
    import sys
    parser = init_argparse()
    args, _ = parser.parse_known_args()
    if len(sys.argv) == 1:
        args = validate(args)
    exit(main(args))
