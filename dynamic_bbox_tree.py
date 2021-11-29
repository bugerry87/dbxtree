
## Build in
from argparse import ArgumentParser
from collections import deque

## Installed
import numpy as np

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
		description="SpatialTree",
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
			help='Encode datapoints to a SpatialTree',
			conflict_handler='resolve',
			parents=parents
			)
	else:
		encode_args = ArgumentParser(
			description='Encode datapoints to a SpatialTree',
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
			help='Decode SpatialTree to datapoints',
			conflict_handler='resolve',
			parents=parents
			)
	else:
		decode_args = ArgumentParser(
			description='Decode SpatialTree to datapoints',
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
	encode.count = 0
	buffer = BitBuffer(compressed, 'wb')
	X = lidar.load(uncompressed, xshape, xtype)[..., :oshape[-1]].astype(np.float32)
	bbox = np.abs(X).max(axis=0).astype(np.float32)
	buffer.write(int.from_bytes(np.array(radius).astype(np.float32).tobytes(), 'big'), 32, soft_flush=True)
	buffer.write(bbox.shape[-1] * 32, 8, soft_flush=True)
	buffer.write(int.from_bytes(bbox.tobytes(), 'big'), bbox.shape[-1] * 32, soft_flush=True)

	bbox = np.repeat(bbox[None,...], len(X), axis=0)
	nodes = np.zeros(len(X), dtype=object)
	shifts = np.zeros(len(X), dtype=object)
	r = np.arange(len(X))

	while len(r):
		flags, idx, inv = np.unique(nodes[r], return_index=True, return_inverse=True)
		flags[...] = 0
		m = bbox >= radius
		dims = np.sum(m[r], axis=-1)
		shifts += 3
		i = np.argsort(bbox[r][idx], axis=-1)[::-1][inv]

		sign = X[r[...,None],i] >= 0

		nodes[r] |= np.packbits(sign, -1, 'little').reshape(-1).astype(int) << shifts[r]

		bbox[r] *= 1 - m[r]*0.5
		X[r[...,None],i] += (1-sign*2) * bbox[r[...,None],i] * m[r[...,None],i]
		bits = np.packbits(sign, -1, 'little').reshape(-1).astype(int)
		bits &= (1<<dims) - 1
		np.bitwise_or.at(flags, inv, 1<<bits)
		m = np.all(~m[r][idx], axis=-1)
		np.bitwise_or.at(m, inv, np.all(np.abs(X[r]) <= radius, axis=-1))
		
		dims = dims[idx]
		flags = (flags*~m)[dims>0]
		dims = dims[dims>0]
		r = r[~m[inv]]

		encode.count += len(flags)

		log(encode.count)

		for flag, dim in zip(flags, 1<<dims):
			buffer.write(flag, dim, soft_flush=True)
	buffer.close()
	log("Done", len(np.unique(nodes)))
	pass


def decode(compressed, uncompressed,
	**kwargs
	):
	"""
	"""
	buffer = BitBuffer(compressed, 'rb')
	radius = np.frombuffer(buffer.read(32).to_bytes(4, 'big'), dtype=np.float32)[0]
	bbox_bits = buffer.read(8)
	bbox = buffer.read(bbox_bits).to_bytes(bbox_bits // 8, 'big')
	bbox = np.frombuffer(bbox, dtype=np.float32)[None,...]
	ii = np.argsort(bbox)[::-1]
	mask = bbox >= radius
	shifts = np.sum(mask, axis=-1)
	X = np.zeros_like(bbox)

	while True:
		flags = np.hstack([buffer.read(1<<shift) for shift in shifts])
		done = flags == 0
		flags = [flag >> np.arange(8) & 1 for flag in flags]
		idx = [np.full(max(np.sum(bits),1),i) for i, bits in enumerate(flags)]
		idx = np.hstack(idx)
		done = done[idx]
		mask = np.hstack(flags) == 1
		tokens = np.arange(8) * np.vstack(flags)
		tokens = np.hstack(tokens)[mask]
		tokens = np.vstack([(token >> i & 1)*2 - 1 for i, token in zip(ii[idx], tokens)])
		bbox = bbox[idx][~done]
		mask = bbox >= radius
		bbox = bbox * (1 - mask*0.5)
		shifts = np.sum(mask, axis=-1)
		shifts = shifts[shifts>0]
		X = X[idx]
		X[~done] += bbox * tokens * mask
		ii = np.argsort(bbox)[::-1]
		print(shifts.shape)
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