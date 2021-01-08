
## Installed
import tensorflow as tf
from tensorflow.keras.metrics import MeanMetricWrapper, top_k_categorical_accuracy


class FlatTopKAccuracy(MeanMetricWrapper):
	"""
	"""
	def __init__(self, k, classes,
		name='flat_topk_accuracy'
		):
		"""
		"""
		super(FlatTopKAccuracy, self).__init__(top_k_categorical_accuracy, name=name, k=k, **kwargs)
		self.classes = classes
		pass

	def update_state(self, y_true, y_pred, sample_weight=None):
		y_true = tf.reshape(y_true, (-1, self.classes))
		y_pred = tf.reshape(y_pred, (-1, self.classes))
		if sample_weight is not None:
			sample_weight = tf.reshape(sample_weight, (-1, self.classes))
		super(FlatTopKAccuracy, self).update_state(y_true, y_pred)
	
	#def call(self, y_true, y_pred, sample_weight):
	#	super(FlatTopKAccuracy, self).call(y_true, y_pred)
	#
	#def __call__(self, y_true, y_pred, sample_weight):
	#	super(FlatTopKAccuracy, self).__call__(y_true, y_pred)