

## Build in
from argparse import ArgumentParser

## Installed
import numpy as np
import tensorflow as tf

## Local
import mhdm.tfops.dbxtree as dbxtree
from mhdm.bitops import BitBuffer
from mhdm.utils import log, ifile


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
		metavar='WILDCARD',
		help='Name of the uncompressed file'
		)
	
	main_args.add_argument(
		'--compressed', '-Y',
		metavar='WILDCARD',
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

if __name__ == '__main__':
	radius = 0.003
	X = np.fromfile('data/0000000000.bin', np.float32).reshape(-1,4)[...,:3]
	bbox = np.abs(X).max(axis=0).astype(np.float32)

	buffer = BitBuffer('data/tf_dyntree.bin', 'wb')
	buffer.write(int.from_bytes(np.array(radius).astype(np.float32).tobytes(), 'big'), 32, soft_flush=True)
	buffer.write(bbox.shape[-1] * 32, 8, soft_flush=True)
	buffer.write(int.from_bytes(bbox.tobytes(), 'big'), bbox.shape[-1] * 32, soft_flush=True)

	i = np.argsort(bbox)[...,::-1]
	BB = bbox = tf.constant(bbox[i])
	X = x = tf.constant(X[...,i])
	pos = tf.zeros_like(bbox)[None, ...]
	nodes = tf.constant(np.ones(len(X), dtype=np.int64))
	F = []
	
	print("Encoding...")
	delta = time_delta()
	next(delta)
	dims = 3
	while dims:
		x, nodes, pivots, pos, bbox, flags, uids, dims = dbxtree.encode(x, nodes, pos, bbox, radius)
		dims = dims.numpy()
		F.append(flags)
	print("Inference Time:", next(delta))
	
	print("Decoding...")
	bbox = BB
	Y = tf.zeros([1,3], dtype=tf.float32)
	keep = tf.zeros([0,3], dtype=tf.float32)
	for flags in F:
		Y, keep, bbox = dynamictree.decode(flags, bbox, radius, Y, keep)
	print("Inference Time:", next(delta))

	print("Evaluation...")
	Xtree = cKDTree(X)
	Ytree = cKDTree(Y)
	XYdelta, XYnn = Xtree.query(Y)
	YXdelta, YXnn = Ytree.query(X)
	print("PSNR XY:", psnr(np.mean(XYdelta**2)))
	print("PSNR YX:", psnr(np.mean(YXdelta**2)))
	Y.numpy().tofile('data/test_dbx_tree.bin')


def encode():
	pass


def decode():
	pass


def main(args, unparsed):
	log.verbose = args.verbose
	args.run(**args.__dict__)


if __name__ == '__main__':
	main_args = init_main_args()
	subparser = main_args.add_subparsers(help='Application Modes:')
	init_encode_args([main_args], subparser)
	init_decode_args([main_args], subparser)
	main(*main_args.parse_known_args())