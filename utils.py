'''
Helper functions for this project.

Author: Gerald Baulig
'''

#Global libs
import numpy as np
import matplotlib.pyplot as plt
from time import time
from glob import glob, iglob


def myinput(prompt, default=None, cast=None):
    ''' myinput(prompt, default=None, cast=None) -> arg
    Handle an interactive user input.
    Returns a default value if no input is given.
    Casts or parses the input immediately.
    Loops the input prompt until a valid input is given.
    
    Args:
        prompt: The prompt or help text.
        default: The default value if no input is given.
        cast: A cast or parser function.
    '''
    while True:
        arg = input(prompt)
        if arg == '':
            return default
        elif cast != None:
            try:
                return cast(arg)
            except:
                print("Invalid input type. Try again...")
        else:
            return arg
    pass


def ifile(wildcards, sort=False, recursive=True):
    def sglob(wc):
        if sort:
            return sorted(glob(wc, recursive=recursive))
        else:
            return iglob(wc, recursive=recursive)

    if isinstance(wildcards, str):
        for wc in sglob(wildcards):
            yield wc
    elif isinstance(wildcards, list):
        if sort:
            wildcards = sorted(wildcards)
        for wc in wildcards:
            if any(('*?[' in c) for c in wc):
                for c in sglob(wc):
                    yield c
            else:
                yield wc
    else:
        raise TypeError("wildecards must be string or list.")  


def arrange_subplots(pltc):
    ''' arrange_subplots(pltc) -> fig, axes
    Arranges a given number of plots to well formated subplots.
    
    Args:
        pltc: The number of plots.
    
    Returns:
        fig: The figure.
        axes: A list of axes of each subplot.
    '''
    cols = int(np.floor(np.sqrt(pltc)))
    rows = int(np.ceil(pltc/cols))
    fig, axes = plt.subplots(cols,rows)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes]) #fix format so it can be used consistently.
    
    return fig, axes.flatten()


last_call = 0
def time_delta():
    ''' time_delta() -> delta
    Captures time delta from last call.
    
    Returns:
        delta: Past time in seconds.
    '''
    global last_call
    delta = time() - last_call
    return delta
 
 
 def polarize(X, scale=(10,10)):
    P = X.copy()
    P[:,0] = np.arccos(X[:,0] / np.linalg.norm(X[:,(0,1)], axis=1)) * (1*(X[:,1] >= 0) - (X[:,1] < 0)) * scale[0]
    P[:,1] = np.linalg.norm(X[:,:2], axis=1)
    P[:,2] = np.arcsin(P[:,2] / P[:,1]) * scale[1]
    return P


def face_normals(T, normalize=True):
    fN = np.cross(T[:,1] - T[:,0], T[:,2] - T[:,0])
    if normalize:
        return fN / np.linalg.norm(fN, axis=1)[:, None]
    else:
        return fN


def edge_normals(fN, Ti_flat, normalize=True):
    fN = fN.repeat(3, axis=0)
    eN = np.zeros((Ti_flat.max()+1, 3))
    for fn, i in zip(fN, Ti_flat):
        eN[i] += fn
    if normalize:
        return eN / np.linalg.norm(eN, axis=1)[:, None]
    else:
        return eN


def dist2line(A, B, P):
    D = np.zeros(A.shape)
    AP = P - A
    BP = P - B
    AB = B - A
    AB_norm = np.linalg.norm(AB, axis=1)[:, None]
    D = np.cross(AP, AB) / AB_norm
    return D