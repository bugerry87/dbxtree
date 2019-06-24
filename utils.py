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