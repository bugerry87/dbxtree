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
import matplotlib.pyplot as plt
import numpy as np
import pykitti  # install using pip install pykitti

#Local libs
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
    
    logs = {'L_mean':{}, 'ACC05':{}, 'ACC01':{}, 'Compress':{}}
    magnitudes = [0.01, 0.05, 0.1, 0.5, 1, 2]
    
    np.random.seed(0)
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

    frames = [next(frames) for i in range(1)]

    for X in frames:
        if not len(X):
            break
        
        print("Input size:", X.shape)
        P = X.copy()[:,:3]
        np.random.shuffle(P)
        P = np.array_split(P, 10)[0]
        print("Query size:", P.shape)
        
        for m in magnitudes:
            logs['L_mean'][m] = []
            logs['ACC01'][m] = []
            logs['ACC05'][m] = []
            logs['Compress'][m] = []
        
        for m in magnitudes:
            print("\nMagnitude:", m)
            Q = quantirize(X[:,:3], m)
            Y = np.arange(Q.shape[0])
            Qi = np.array((range(Q.shape[0]-1), range(1,Q.shape[0]))).T
            print("Model size:", Q.shape)
            
            print("Compute loss...")
            tree = KDNTree(Q, Qi, j=8, leaf_size=100)
            print("\n0%                      |50%                     |100%")
            L, mp, nn = tree.query(P, callback=callback)
            
            L_mean = np.sqrt(L.mean())
            print("\nLoss mean:", L_mean)
            
            logs['L_mean'][m].append(L_mean)
            logs['ACC01'][m].append(np.mean(L<=0.1**2))
            logs['ACC05'][m].append(np.mean(L<=0.5**2))
            logs['Compress'][m].append(Q.shape[0] / X.shape[0])
    
    L_mean = [np.mean(logs['L_mean'][m]) for m in magnitudes]
    ACC01 = [np.mean(logs['ACC01'][m]) for m in magnitudes]
    ACC05 = [np.mean(logs['ACC05'][m]) for m in magnitudes]
    Compress = [np.mean(logs['Compress'][m]) for m in magnitudes]
    
    fig = plt.figure()
    fig.subplots_adjust(top=0.8)
    ax = fig.add_subplot(111)
    ax.set_xlabel('magnitude (m)')
    
    ax.plot(magnitudes, L_mean, label='L_mean')
    ax.plot(magnitudes, ACC01, label='ACC01')
    ax.plot(magnitudes, ACC05, label='ACC05')
    ax.plot(magnitudes, Compress, label='Compress')
    
    plt.legend()
    plt.show()
    return 0

if __name__ == '__main__':
    import sys
    parser = init_argparse()
    args, _ = parser.parse_known_args()
    if len(sys.argv) == 1:
        args = validate(args)
    exit(main(args))
