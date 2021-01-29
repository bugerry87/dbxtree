
## Installed
import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy, MSLE
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.python.keras.losses import LossFunctionWrapper


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


def regularized_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0, msle_smooting=1.e-8):
	#y_pred = tf.convert_to_tensor(y_pred)
	#y_true = tf.cast(y_true, y_pred.dtype)
	cc = categorical_crossentropy(y_true, y_pred, from_logits, label_smoothing)
	msle = MSLE(y_pred, y_true)
	#reg = tf.math.reduce_max(y_pred) - tf.math.reduce_max(y_true)
	#reg = 1 - tf.math.exp(-reg**2)
	return cc + msle * msle_smooting


class RegularizedCrossentropy(LossFunctionWrapper):
	"""
	"""
	def __init__(self,
		name='regularized_crossentropy',
		from_logits=False,
		label_smoothing=0
		):
		"""
		"""
		super(RegularizedCrossentropy, self).__init__(
			regularized_crossentropy,
			name=name,
			from_logits=from_logits,
			label_smoothing=label_smoothing
			)
		pass