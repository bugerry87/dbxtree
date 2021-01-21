
## Installed
import tensorflow as tf
from tensorflow.keras.metrics import TopKCategoricalAccuracy


class FlatTopKAccuracy(TopKCategoricalAccuracy):
	"""
	"""
	def __init__(self, k, classes,
		name='flat_topk_accuracy',
		**kwargs
		):
		"""
		"""
		super(FlatTopKAccuracy, self).__init__(name=name, k=k, **kwargs)
		self.classes = classes
		pass

	def update_state(self, y_true, y_pred, sample_weight=None):
		y_true = tf.reshape(y_true, (-1, self.classes))
		y_pred = tf.reshape(y_pred, (-1, self.classes))
		return super(FlatTopKAccuracy, self).update_state(y_true, y_pred)