

## Build in
from argparse import ArgumentParser

## Installed
import numpy as np
import tensorflow as tf

## Local
import mhdm.tfops.dbxtree as dbxtree
from mhdm.bitops import BitBuffer
from mhdm.utils import log, ifile, time_delta


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


def encode(
	uncompressed,
	compressed,
	radius = 0.03,
	xtype = 'float32',
	xshape = (-1,4),
	oshape = (-1,3),
	**kwargs
	):
	"""
	"""
	log("Encoding...")
	c = 0
	files = [f for f in ifile(uncompressed)]
	buffer = BitBuffer()
	delta_total = time_delta()
	next(delta_total)
	for f in files:
		outname = f"{compressed}.{c:05d}.dbx.bin" if len(files) > 1 else f"{compressed}.dbx.bin"
		X = np.fromfile(f, xtype).reshape(*xshape)[...,:oshape[-1]]
		bbox = np.abs(X).astype(np.float32).max(axis=0)

		buffer.open(outname, 'wb')
		buffer.write(int.from_bytes(np.array(radius).astype(np.float32).tobytes(), 'big'), 32, soft_flush=True)
		buffer.write(bbox.shape[-1] * 32, 8, soft_flush=True)
		buffer.write(int.from_bytes(bbox.tobytes(), 'big'), bbox.shape[-1] * 32, soft_flush=True)

		i = np.argsort(bbox)[...,::-1]
		BB = bbox = tf.constant(bbox[i])
		X = tf.constant(X[...,i])
		pos = tf.zeros_like(bbox)[None, ...]
		nodes = tf.ones(len(X), dtype=np.int64)
		
		delta = time_delta()
		next(delta)
		dims = 3
		log(f"{f} -> {outname} ", end="")
		while dims:
			X, nodes, pivots, pos, bbox, flags, uids, dims = dbxtree.encode(X, nodes, pos, bbox, radius)
			dims = dims.numpy()
			log(end=".", flush=True)
			if dims:
				for flag in flags.numpy():
					buffer.write(flag, 1<<dims, soft_flush=True)
		buffer.close()
		log(f" {next(delta)}s")
		c += 1
	log(f"Done in {next(delta_total)}s")


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