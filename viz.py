#!/usr/bin/env python
"""
VISUALISE THE LIDAR DATA FROM THE ANY DATASET

Based on the sample code from
    https://github.com/utiasSTARS/pykitti/blob/master/demos/demo_raw.py
And:
    http://stackoverflow.com/a/37863912
    
Author: Gerald Baulig
"""

from mayavi import mlab
from mayavi.core.ui.api import MlabSceneModel, SceneEditor
from traits.api import HasTraits
from traitsui.api import View, Item, Group
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA

class GUI(HasTraits):
    def __init__(self, items=None):
        self.fig = Instance(MlabSceneModel, ())
        self.view = View(
            Group(
                Item('scene1',
                    editor=SceneEditor(), height=250,
                    width=300),
                'button1',
                show_labels=False
                ),
            resizable=True
            )


def create_figure():
    return mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))


def clear_figure(fig):
    mlab.clf(fig)


def mesh(X, T, Y, fig, plot=None):
    if not len(X):
        raise ValueError("Error: Empty frame!")

    if plot == None:
        plot = mlab.triangular_mesh(
            X[:,0],
            X[:,1],
            X[:,2],
            T,
            scalars=Y,
            colormap='spectral',  # 'bone', 'copper',
            line_width=10,        # Scale of the line, if any
            figure=fig,
            )
    else:
        plot.mlab_source.reset(
            x = X[:,0],
            y = X[:,1],
            z = X[:,2],
            triangles = T,
            scalars = Y
        )
    return plot


def normals():
    X = X[T[:,0]]
    mlab.quiver3d(
        X[:,0],
        X[:,1],
        X[:,2],
        N[:,0],
        N[:,1],
        N[:,2],
        scale_factor=0.1)


def vertices(X, Y, fig, plot=None):
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
            figure=fig,
            )
    else:
        plot.mlab_source.reset(
            x = X[:, 0],
            y = X[:, 1],
            z = X[:, 2],
            scalars = Y
        )
    return plot


def lines(X, Y, fig, plot=None):
    if not len(X):
        raise ValueError("Error: Empty frame!")

    if plot == None:
        plot = mlab.plot3d(
            X[:,0],
            X[:,1],
            X[:,2],
            Y,
            colormap='spectral',  # 'bone', 'copper',
            tube_radius=None,
            figure=fig,
            )
    else:
        plot.mlab_source.reset(
            x = X[:, 0],
            y = X[:, 1],
            z = X[:, 2],
            scalars = Y
        )
    return plot


if __name__ == '__main__' or 'PLOT_MAIN' in globals():
    #Standard libs
    import sys
    from argparse import ArgumentParser

    #3rd-Party libs
    import numpy as np
    import pykitti

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
        fig = create_figure()
        pca = PCA(3)

        @mlab.animate(delay=10)
        def animation():
            p1 = None
            p2 = None
            X = next(frames, [])
            while len(X):
                pca.fit(X[:,(1,0,3)])
                twist = pca.components_
                Xtwist = pca.transform(X[:,:3])
                p1 = vertices(X, X[:,3], fig, p1)
                p2 = vertices(Xtwist, -X[:,3], fig, p2)
                r = R.from_matrix(pca.components_)
                print(r.as_rotvec())
                
                
                fig.render() 
                yield
                X = next(frames, [])

        animator = animation()
        mlab.show(stop=False)
        return 0


    parser = init_argparse()
    args, _ = parser.parse_known_args()
    if len(sys.argv) == 1:
        args = validate(args)
    exit(main(args))