from mhdm.tfops.layers import tf, OuterTransformer


def test_outer_transformer_runable():
	transformer = OuterTransformer(layer_type=None)
	A = tf.random.normal([1,1000,8])
	B = tf.random.normal([1,100000,8])
	y = transformer([A, B, A])
	assert(y.shape == [1,100000,8])
	
