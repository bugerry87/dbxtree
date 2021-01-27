from mhdm.tfops.layers import tf, OuterTransformer


def test_outer_transformer_runable():
	transformer = OuterTransformer(8)
	x = tf.random.normal([1,100,3])
	y = transformer(x)
	assert(y.shape == [1,100,8])
	
