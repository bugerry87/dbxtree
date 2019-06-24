'''
A set of kernel functions and helpers.

Author: Gerald Baulig
'''

#Global libs
import numpy as np

KERNELS = ['euclidean', 'linear', 'RBF', 'linearRBF', 'none']
none = None


def parse_params(arg_string):
    ''' parse_params(arg_string) -> dict
    Parses an argument string to a dictionary.
    E.g.: 'gamma=1.0, sigma=2.0'
    
    Args:
        arg_string: A comma seperated argument string.
    Returns:
        params: The related dictionary of key=value.
    '''
    if arg_string:
        params = arg_string.strip()
        params = params.split(',')
        params = [p.split('=') for p in params]
        params = {p[0]:p[1] for p in params}
        return params
    else:
        return {}


def euclidean(u, v, params={}):
    ''' linear(u, v, params={}) -> G
    A simple euclidean kernel.
    Calculates the distance of each pairing.
        k(u,v) = (u - v.T)^2
    
    Args:
        u: The left hand values.
        v: The right hand values.
        params: No params required.
    Returns:
        G: The Gram-Matrix of a linear kernel
    '''
    if len(u.shape) == 1:
        u = u[:,None]

    if len(v.shape) == 1:
        v = v[:,None]
    
    return np.sum(u**2, axis=1) + np.sum(v**2, axis=1)[:,None] - 2*np.dot(u,v.T)


def linear(u, v, params={}):
    ''' linear(u, v, params={}) -> G
    A simple linear kernel.
        k(u,v) = u * v.T
    
    Args:
        u: The left hand values.
        v: The right hand values.
        params: No params required.
    Returns:
        G: The Gram-Matrix of a linear kernel
    '''
    if len(u.shape) == 1:
        u = u[:,None]

    if len(v.shape) == 1:
        v = v[:,None]
    return np.dot(u, v.T)


def RBF(u, v, params={}):
    ''' RBF(u, v, params={}) -> G
    A simple Radial Basis Function kernel.
        k(u,v) = exp(-g*(u - v.T)^2)
    
    Args:
        u: The left hand values.
        v: The right hand values.
        params:
            gamma: The variance factor.
    Returns:
        G: The Gram-Matrix of an RBF kernel
    '''
    g = float(params['gamma']) if 'gamma' in params else 1.0
    return np.exp(-g * euclidean(u,v))


def linearRBF(u, v, params={}):
    ''' RBF(u, v, params={}) -> G
    A combination of linear and Radial Basis Function kernel.
        k(u,v) = u * v.T + exp(-g*(u - v.T)^2)
    
    Args:
        u: The left hand values.
        v: The right hand values.
        params:
            gamma: The variance factor.
    Returns:
        G: The Gram-Matrix of a linearRBF kernel
    '''
    return linear(u, v) + RBF(u, v, params)
