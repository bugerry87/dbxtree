
## Installed
import tensorflow as tf
from tensorflow.keras.layers import Reshape
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.metrics import Metric


def focal_loss(y_true, y_pred,
	gamma=5.0,
	**kwargs
	):
	"""
	"""
	pt = (1.0 - y_true) - y_pred * (1.0 - y_true * 2.0)
	loss = -(1 - pt) ** gamma * tf.math.log(pt)
	return tf.math.reduce_mean(loss)


def combinate(y_true, y_pred, loss_funcs):
	def parse(loss_funcs):
		for loss_func in loss_funcs:
			if 'slices' in loss_func:
				gt = y_true[...,loss_func.slices]
				est = y_pred[...,loss_func.slices]
			else:
				gt = y_true
				est = y_pred
			
			if 'reshape' in loss_func:
				reshape = Reshape(loss_func.reshape)
				gt =  reshape(gt)
				est = reshape(est)
			
			weights = loss_func.weights if 'weights' in loss_func else 1.0
			kwargs = loss_func.kwargs if 'kwargs' in loss_func else dict()

			if 'loss' in loss_func:
				yield loss_func.loss(gt, est, **kwargs) * weights
			else:
				yield loss_func(gt, est, **kwargs) * weights
		pass

	loss = tf.math.reduce_sum([*parse(loss_funcs)])
	return loss


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


class SlicedMetric(Metric):
	"""
	"""
	def __init__(self, metric, slices, **kwargs):
		super(SlicedMetric, self).__init__(name='sliced_' + metric.name, **kwargs)
		self.slices = slices
		self.metric = metric
		pass

	def merge_state(self, metrics):
		return self.metric.merge_state(metrics)

	def update_state(self, y_true, y_pred, sample_weights=None):
		return self.metric.update_state(
			y_true[...,self.slices], 
			y_pred[...,self.slices], 
			sample_weights[...,self.slices] if sample_weights is not None else None
		)
	
	def result(self):
		return self.metric.result()
	
	def reset_result(self):
		return self.metric.reset_result()