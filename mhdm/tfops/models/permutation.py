
## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.python.keras.engine import data_adapter

## Local
from .. import count
from .. import bitops

def permute(X, bits):
    """
    """
    def cond(*args):
        return True

    def body(X, P, idx):
        num_seg = tf.math.reduce_max(idx) + 1
        p = tf.math.unsorted_segment_sum(X, idx, num_seg) #n,b
        p = tf.concat([p[...,None], num_points - p[...,None]]) #n,b,2
        p = tf.math.reduce_max(p, axis=-1) #n,b
        pid = tf.argmax(p, axis=-1)[idx, None]

        mask = pid != tf.range(X.shape[-1], dtype=X.dtype)
        mask = tf.reshape(mask, -1)
        x = X[pid, None]
        X = tf.reshape(X[mask], X.shape[-1] - 1)
        idx = bitops.left_shift(idx, 1) + x
        argsort = tf.argsort(idx)
        argsort = tf.range(num_points)[argsort]

        p = p[argsort]
        P = tf.concat([P, p], axis=-1)
        idx = tf.unique(idx, X.dtype)[-1]
        return X, P, idx

    num_points = count(X)
    shifts = tf.range(bits, dtype=X.dtype)
    one = tf.constant(1, dtype=X.dtype)
    X = bitops.right_shift(X, shifts)
    X = bitops.bitwise_and(X, one)
    P = tf.zeros((num_points, 0), X.dtype)
    idx = tf.zeros_like(X)
    
    P = tf.while_loop(
        cond, body,
        loop_vars=(X, P, idx),
        shape_invariants=(tf.TensorShape((None,None)), tf.TensorShape((None,None)), idx.shape),
        maximum_iterations=bits-1
        )[1]
    return P


class Permutation(Model):
    """
    """
    def __init__(self):
        """
        """

        self.encoder = Sequential()
        self.decoder = Sequential()
        pass

    def dataloader(self):

        #serialize
        #permute
        pass

    def call(self, X):
        X = self.encoder(X)
        X = self.decoder(X)
        return X


    