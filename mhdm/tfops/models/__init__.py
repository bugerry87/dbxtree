## Installed
import tensorflow as tf

@tf.function
def normalize(X):
	n = tf.stop_gradient(X)
	n = tf.abs(n)
	n = tf.math.reduce_max(n, axis=-1, keepdims=True)
	X = tf.math.divide_no_nan(X, n)
	return X

## Inhire
from .nbittree import NbitTree
from .dynamictree import DynamicTree
from .entropymap import EntropyMapper

__all__ = [NbitTree, EntropyMapper, DynamicTree, normalize]