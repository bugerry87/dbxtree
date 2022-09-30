## Installed
import tensorflow as tf

@tf.function
def normalize(X):
	n = tf.abs(X)
	n = tf.math.reduce_max(n, axis=-1, keepdims=True)
	X = tf.math.divide_no_nan(X, n)
	return X

def batching(db, window_size, batch_size=1):
	@tf.function
	def padding(*args):
		return [tf.pad(arg, (window_size//2, 0)) for arg in args]
	
	db = db.map(padding)
	db = db.window(window_size, 1, 1)
	db = db.batch(batch_size)
	return db

## Inhire
from .nbittree import NbitTree
from .dbxtree import DynamicTree
from .dbxtree2 import DynamicTree2
from .entropymap import EntropyMapper

__all__ = [NbitTree, EntropyMapper, DynamicTree, DynamicTree2, normalize]