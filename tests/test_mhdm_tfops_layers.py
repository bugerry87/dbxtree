from mhdm.tfops.layers import tf, OuterTransformer


def test_outer_transformer_runable():
	transformer = OuterTransformer(layer_type=None)
	A = tf.random.normal([1,100,3])
	B = tf.random.normal([1,10,3])
	y = transformer([A, B, B])
	assert(y.shape == [1,100,3])
	
