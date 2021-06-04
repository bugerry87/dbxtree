
## Installed
import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy, cosine_similarity, MSLE
from tensorflow.python.keras.losses import LossFunctionWrapper


def regularized_crossentropy(y_true, y_pred, 
	from_logits=False,
	label_smoothing=0,
	msle_smoothing=1.0,
	slices=None,
	reshape=None
	):
	"""
	"""
	if slices is not None:
		y_true = y_true[...,slices[0]:slices[1]]
		y_pred = y_pred[...,slices[0]:slices[1]]
	
	if reshape is not None:
		y_true = tf.reshape(y_true, (-1, *reshape))
		y_pred = tf.reshape(y_pred, (-1, *reshape))

	cc = categorical_crossentropy(y_true, y_pred, from_logits, label_smoothing)
	msle = MSLE(y_true, y_pred)
	loss = cc + msle * msle_smoothing
	if reshape is not None:
		loss = tf.math.reduce_sum(loss, axis=-1)[None,...]
	return loss


def regularized_cosine(y_true, y_pred,
	msle_smoothing=1.0,
	slices=None,
	reshape=None
	):
	"""
	"""
	if slices is not None:
		y_true = y_true[...,slices[0]:slices[1]]
		y_pred = y_pred[...,slices[0]:slices[1]]
	
	if reshape is not None:
		y_true = tf.reshape(y_true, (-1, *reshape))
		y_pred = tf.reshape(y_pred, (-1, *reshape))

	cs = cosine_similarity(y_true, y_pred)
	msle = MSLE(y_true, y_pred)
	loss =  cs + msle * msle_smoothing + 1.0
	if reshape is not None:
		loss = tf.math.reduce_sum(loss, axis=-1)[None,...]
	return loss


def combinate(y_true, y_pred, loss_funcs, loss_kwargs, loss_weights):
	loss = tf.math.reduce_sum([loss(y_true, y_pred, **kwargs) * weight for loss, kwargs, weight in zip(loss_funcs, loss_kwargs, loss_weights)], axis=0)
	return loss


class RegularizedCrossentropy(LossFunctionWrapper):
	"""
	"""
	def __init__(self,
		name='regularized_crossentropy',
		**kwargs
		):
		"""
		"""
		super(RegularizedCrossentropy, self).__init__(
			regularized_crossentropy,
			name=name,
			**kwargs
			)
		pass


class RegularizedCosine(LossFunctionWrapper):
	"""
	"""
	def __init__(self,
		name='regularized_cosine',
		**kwargs
		):
		"""
		"""
		super(RegularizedCosine, self).__init__(
			regularized_cosine,
			name=name,
			**kwargs
			)
		pass


class CombinedLoss(LossFunctionWrapper):
	"""
	"""
	def __init__(self, loss_funcs, loss_kwargs, loss_weights,
		name='combined_loss',
		**kwargs
		):
		"""
		"""
		super(CombinedLoss, self).__init__(
			combinate,
			name=name,
			loss_funcs = loss_funcs,
			loss_kwargs = loss_kwargs,
			loss_weights = loss_weights,
			**kwargs
			)
		pass