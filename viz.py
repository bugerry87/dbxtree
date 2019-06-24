"""
VISUALISE THE LIDAR DATA FROM THE KITTI DATASET

Based on the sample code from
    https://github.com/utiasSTARS/pykitti/blob/master/demos/demo_raw.py
And:
    http://stackoverflow.com/a/37863912
"""

#Global libs
import pykitti  # install using pip install pykitti
import numpy as np
from mayavi import mlab
from glob import iglob

#Local libs
import cluster

# Load the data
files = iglob('./2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/data/*.bin')
frames = pykitti.utils.yield_velo_scans(files)
fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))

frame = next(frames, [])
if not len(frame):
    print("Error: Empty frame!")
    exit(1)


'''
idx = frame[:,2] > np.mean(frame[:,2])-0.2
frame = frame[idx]

means = cluster.init_kmeans(frame, 100, mode='kmeans++')
for Y, means, _, _ in cluster.kmeans(frame, means):
    pass
'''

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

@mlab.animate(delay=10)
def init_animation():
    global frames, fig, plot
    frame = next(frames, [])
    while len(frame):
        plot.mlab_source.reset(
            x = frame[:, 0],
            y = frame[:, 1],
            z = frame[:, 2],
            scalars = frame[:, 3]
        )
        fig.render()
        yield
        frame = next(frames, [])

input("Hit any key to continue!")
animator = init_animation()

mlab.show()

