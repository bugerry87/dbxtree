
## Build in
from argparse import ArgumentParser
from collections import deque

## Installed
import numpy as np

## Local
from mhdm.utils import log, ifile
from mhdm.bitops import BitBuffer
import mhdm.spatial as spatial
import mhdm.bitops as bitops
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
	def expand(X, bbox):
		i = np.argsort(bbox)[::-1]
		i = i[bbox[i] >= radius]
		dim = len(i)
		flag_size = 1<<dim
		if dim == 0:
			encode.count += 1
			log("BBox:", bbox, "bits:", i, "Points Detected:", encode.count)
			return
		if np.all(np.all(np.abs(X) <= radius, axis=-1)):
			flags.write(0, flag_size, soft_flush=True)
			encode.count += 1
			log("BBox:", bbox, "bits:", i, "Points Detected:", encode.count)
			return
		
		m = X[...,i] >= 0
		bbox[...,i] *= 0.5
		X[...,i] += (1 - m*2) * bbox[...,i]

		flag = 0
		t = np.packbits(m, -1, 'little').reshape(-1)
		for d in range(flag_size):
			m = t==d
			if np.any(m):
				flag |= 1<<d
				yield expand(X[m], bbox.copy())
		flags.write(flag, flag_size, soft_flush=True)
	
	encode.count = 0
	flags = BitBuffer(compressed, 'wb')
	X = lidar.load(uncompressed, xshape, xtype)[..., :oshape[-1]].astype(np.float32)
	bbox = np.abs(X).max(axis=0).astype(np.float32)
	flags.write(int.from_bytes(np.array(radius).astype(np.float32).tobytes(), 'big'), 32, soft_flush=True)
	flags.write(bbox.shape[-1] * 32, 8, soft_flush=True)
	flags.write(int.from_bytes(bbox.tobytes(), 'big'), bbox.shape[-1] * 32, soft_flush=True)
	nodes = deque(expand(X, bbox))
	while nodes:
		node = nodes.popleft()
		nodes.extend(node)
	
	flags.close()
	log("Done")
	pass


def decode(compressed, uncompressed,
	**kwargs
	):
	"""
	"""
	def expand(x, bbox, i):
		dim = len(i)
		flag_size = 1<<dim
		if dim == 0:
			X.append(x)
			decode.count += 1
			log("X:", x, "bits:", i, "Points Detected:", decode.count)
			return
		
		flag = flags.read(flag_size)
		if not flag:
			X.append(x)
			decode.count += 1
			log("X:", x, "bits:", i, "Points Detected:", decode.count)
			return

		bbox[...,i] *= 0.5
		for d in np.arange(flag_size, dtype=np.uint8):
			if not flag >> d & 1: continue
			m = np.unpackbits(d, -1, dim, 'little').astype(np.float32)
			xx = x.copy()
			xx[...,i] -= (1 - m*2) * bbox[...,i]
			ii = np.argsort(bbox)[::-1]
			ii = ii[bbox[ii] >= radius]
			yield expand(xx, bbox.copy(), ii)
	
	decode.count = 0
	flags = BitBuffer(compressed, 'rb')
	radius = np.frombuffer(flags.read(32).to_bytes(4, 'big'), dtype=np.float32)[0]
	bbox_bits = flags.read(8)
	bbox = flags.read(bbox_bits).to_bytes(bbox_bits // 8, 'big')
	bbox = np.frombuffer(bbox, dtype=np.float32)
	i = np.argsort(bbox)[::-1]
	X = []
	nodes = deque(expand(np.zeros_like(bbox), bbox.copy(), i))
	while nodes:
		node = nodes.popleft()
		nodes.extend(node)
	
	lidar.save(np.vstack(X), uncompressed)
	flags.close()
	log("Done")
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