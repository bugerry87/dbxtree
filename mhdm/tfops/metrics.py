
## Installed
import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy, cosine_similarity, MSLE
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.python.keras.losses import LossFunctionWrapper


def regularized_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0, msle_smoothing=1.0):
	"""
	"""
	cc = categorical_crossentropy(y_true, y_pred, from_logits, label_smoothing)
	msle = MSLE(y_true, y_pred)
	return cc + msle * msle_smoothing


def regularized_cosine(y_true, y_pred, msle_smoothing=1.0):
	"""
	"""
	cs = cosine_similarity(y_true, y_pred)
	msle = MSLE(y_true, y_pred)
	return cs + msle * msle_smoothing + 1.0


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


class RegularizedCrossentropy(LossFunctionWrapper):
	"""
	"""
	def __init__(self,
		name='regularized_crossentropy',
		from_logits=False,
		label_smoothing=0,
		msle_smoothing=1.0
		):
		"""
		"""
		super(RegularizedCrossentropy, self).__init__(
			regularized_crossentropy,
			name=name,
			from_logits=from_logits,
			label_smoothing=label_smoothing,
			msle_smoothing=msle_smoothing
			)
		pass


class RegularizedCosine(LossFunctionWrapper):
	"""
	"""
	def __init__(self,
		name='regularized_cosine',
		msle_smoothing=1.0
		):
		"""
		"""
		super(RegularizedCosine, self).__init__(
			regularized_cosine,
			name=name,
			msle_smoothing=msle_smoothing
			)
		pass