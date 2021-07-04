## Installed
import tensorflow as tf

@tf.function
def normalize(X):
	n = tf.stop_gradient(X)
	n = tf.abs(X)
	n = tf.math.reduce_max(X, axis=-1, keepdims=True)
	X = tf.math.divide_no_nan(X, n)
	return X

## Inhire
from .nbittree import NbitTree
from .entropymap import EntropyMapper

__all__ = [NbitTree, EntropyMapper, normalize]