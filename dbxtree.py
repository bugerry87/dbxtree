
## Build in
from argparse import ArgumentParser

## Installed
import numpy as np
from sklearn.decomposition import PCA

## Local
from mhdm.utils import log
from mhdm.bitops import BitBuffer
import mhdm.lidar as lidar


def init_main_args(parents=[]):
	"""
	Initialize an ArgumentParser for this module.
	
	Args:
		parents: A list of ArgumentParsers of other scripts, if there are any.
		
	Returns:
		main_args: The ArgumentParsers.
	"""
	main_args = ArgumentParser(
		description="DBXTree",
		conflict_handler='resolve',
		parents=parents
		)
	
	main_args.add_argument(
		'--verbose', '-v',
		action='store_true'
		)
	
	main_args.add_argument(
		'--uncompressed', '-X',
		metavar='FILE',
		help='Name of the uncompressed file'
		)
	
	main_args.add_argument(
		'--compressed', '-Y',
		metavar='FILE',
		help='Name of the compressed file'
		)
	
	main_args.add_argument(
		'--radius', '-r',
		metavar='FLOAT',
		type=float,
		default=0.03,
		help='Accepted radius of error'
		)
	
	main_args.set_defaults(
		run=lambda **kwargs: main_args.print_help()
		)
	return main_args


def init_encode_args(parents=[], subparser=None):
	if subparser:
		encode_args = subparser.add_parser('encode',
			help='Encode datapoints to a DBXTree',
			conflict_handler='resolve',
			parents=parents
			)
	else:
		encode_args = ArgumentParser(
			description='Encode datapoints to a DBXTree',
			conflict_handler='resolve',
			parents=parents
			)
	
	encode_args.add_argument(
		'--xtype', '-t',
		metavar='TYPE',
		default='float32',
		help='The expected data-type of the datapoints (datault=float32)'
		)
	
	encode_args.add_argument(
		'--xshape',
		nargs='+',
		metavar='SHAPE',
		default=(-1,4),
		help='Dimensionality of the input shape'
		)
	
	encode_args.add_argument(
		'--oshape',
		nargs='+',
		metavar='SHAPE',
		default=(-1,3),
		help='Dimensionality of the output shape'
		)
	
	encode_args.set_defaults(
		run=encode
		)
	return encode_args


def init_decode_args(parents=[], subparser=None):
	if subparser:
		decode_args = subparser.add_parser('decode',
			help='Decode DBXTree to datapoints',
			conflict_handler='resolve',
			parents=parents
			)
	else:
		decode_args = ArgumentParser(
			description='Decode DBXTree to datapoints',
			conflict_handler='resolve',
			parents=parents
			)
	
	decode_args.set_defaults(
		run=decode
		)
	return decode_args


def encode(uncompressed, compressed,
	radius=0.03,
	xshape=(-1,4),
	xtype='float32',
	oshape=(-1,3),
	**kwargs
	):
	"""
	"""
	buffer = BitBuffer(compressed, 'wb')
	X = lidar.load(uncompressed, xshape, xtype)[..., :oshape[-1]].astype(xtype)

	pca = PCA(n_components=3)
	pca.fit(X)
	X -= pca.mean_
	X = X@np.linalg.inv(pca.components_)
	print(np.abs(X).max(axis=0))

	bbox = np.abs(X).max(axis=0).astype(xtype)
	buffer.write(int.from_bytes(np.array(radius).astype(xtype).tobytes(), 'big'), 32, soft_flush=True)
	buffer.write(int.from_bytes(bbox.tobytes(), 'big'), bbox.shape[-1] * 32, soft_flush=True)
	buffer.write(int.from_bytes(pca.mean_.astype(xtype).tobytes(), 'big'), pca.mean_.size * 32, soft_flush=True)
	buffer.write(int.from_bytes(pca.components_.tobytes(), 'big'), pca.components_.size * 32, soft_flush=True)

	nodes = np.ones(len(X), dtype=object)
	dims = 3

	while dims:
		u, idx, inv = np.unique(nodes, return_index=True, return_inverse=True)
		flags = np.zeros(len(u), dtype=int)

		big = bbox > radius
		dims = np.sum(big)
		keep = np.zeros(len(u), dtype=bool)
		np.bitwise_or.at(keep, inv, np.any(np.abs(X) > radius, axis=-1))
		keep = keep[inv]
		X = X[keep]
		nodes = nodes[keep]

		sign = X >= 0
		bits = np.packbits(sign, -1, 'little').reshape(-1).astype(int)
		bits &= (1<<dims) - 1
		nodes <<= dims
		nodes |= bits
		np.bitwise_or.at(flags, inv[keep], 1<<bits)

		bbox *= 0.5
		X += (1-sign*2) * bbox * (big > 0)
		args = np.argsort(nodes)
		nodes = nodes[args]
		flags = flags[args]
		X = X[args]

		log(len(X))
		if dims:
			for flag in flags:
				buffer.write(flag, 1<<dims, soft_flush=True)
		pass

	buffer.close()
	log("Done:", len(nodes))
	pass


def decode(compressed, uncompressed,
	**kwargs
	):
	"""
	"""
	buffer = BitBuffer(compressed, 'rb')
	radius = np.frombuffer(buffer.read(32).to_bytes(4, 'big'), dtype=np.float32)[0]
	bbox = buffer.read(3*32).to_bytes(3*4, 'big')
	bbox = np.frombuffer(bbox, dtype=np.float32)
	mean = buffer.read(3*32).to_bytes(3*4, 'big')
	mean = np.frombuffer(mean, dtype=np.float32)[None,...]
	pca = buffer.read(9*32).to_bytes(9*4, 'big')
	pca = np.frombuffer(pca, dtype=np.float32).reshape(3,3)

	X = np.zeros_like(bbox[None,...])
	X_done = np.zeros((0,bbox.shape[-1]), dtype=X.dtype)

	while True:
		mask = bbox > radius
		dims = np.sum(bbox > radius)
		if not dims:
			break
		
		flags = np.hstack([buffer.read(1<<dims) for x in X])
		done = flags == 0
		flags = np.vstack([flag >> np.arange(8) & 1 for flag in flags])
		idx = np.where(flags)
		tokens = 0.5 - (idx[-1][...,None] >> np.arange(3) & 1)
		
		X_done = np.vstack([X_done, X[done]])
		X = X[idx[0]] - bbox * tokens * mask
		bbox = bbox * 0.5
		log(X.shape, X_done.shape)
		pass

	X = np.vstack([X, X_done]).astype(np.float32)
	print(X.max(axis=0))
	X = X@pca
	X += mean
	lidar.save(X, uncompressed)

	buffer.close()
	log("Done:", X, X.shape)


def main(args, unparsed):
	log.verbose = args.verbose
	args.run(**args.__dict__)

if __name__ == '__main__':
	main_args = init_main_args()
	subparser = main_args.add_subparsers(help='Application Modes:')
	init_encode_args([main_args], subparser)
	init_decode_args([main_args], subparser)
	main(*main_args.parse_known_args())