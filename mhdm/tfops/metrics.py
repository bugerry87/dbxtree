
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


def focal_loss(y_true, y_pred, from_logits=False, label_smoothing=0, gamma=5.0):
	pt = (1.0 - y_true) - y_pred * (1.0 - y_true * 2.0)
	loss = -(1 - pt) ** gamma * tf.math.log(pt)
	return tf.math.reduce_mean(loss)


def combinate(y_true, y_pred, loss_funcs):
	def parse(loss_funcs):
		for loss_func in loss_funcs:
			if hasattr(loss_func, 'indices'):
				gt = y_true[...,loss_func.indices]
				est = y_pred[...,loss_func.indices]
			else:
				gt = y_true
				est = y_pred
			
			if hasattr(loss_func, 'weights'):
				weights = loss_func.weights
			else:
				weights = 1
			
			if hasattr(loss_func, 'kwargs'):
				kwargs = loss_func.kwargs
			else:
				kwargs = {}

			if hasattr(loss_func, 'loss'):
				yield loss_func.loss(gt, est, **loss_func.kwargs) * weights
			else:
				yield loss_func(gt, est)
		pass

	loss = tf.math.reduce_sum(parse(loss_funcs), axis=0)
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


class FocalLoss(LossFunctionWrapper):
	"""
	"""
	def __init__(self,
		name='focal_loss',
		**kwargs
		):
		"""
		"""
		super(FocalLoss, self).__init__(
			focal_loss,
			name=name,
			**kwargs
			)
		pass


class CombinedLoss(LossFunctionWrapper):
	"""
	"""
	def __init__(self, loss_funcs,
		name='combined_loss',
		**kwargs
		):
		"""
		"""
		super(CombinedLoss, self).__init__(
			combinate,
			name=name,
			loss_funcs = loss_funcs,
			**kwargs
			)
		pass