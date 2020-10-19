#!/usr/bin/env python3

## Build in
from argparse import ArgumentParser
from collections import deque
import os.path as path
import pickle

## Installed
import numpy as np

## Local
import mhdm.dynamictree as dynamictree
import mhdm.bitops as bitops
from mhdm.utils import Prototype, log, ifile
from mhdm.bitops import BitBuffer


def init_main_args(parents=[]):
	"""
	Initialize an ArgumentParser for this module.
	
	Args:
		parents: A list of ArgumentParsers of other scripts, if there are any.
		
	Returns:
		main_args: The ArgumentParsers.
	"""
	main_args = ArgumentParser(
		description="DynamicTree",
		conflict_handler='resolve',
		parents=parents
		)
	
	main_args.add_argument(
		'--verbose', '-v',
		action='store_true'
		)
	
	main_args.add_argument(
		'--output', '-o',
		metavar='PATH',
		default=None,
		help='A filename for the output data'
		)
	
	main_args.set_defaults(
		run=lambda **kwargs: main_args.print_help()
		)
	
	return main_args


def init_compress_args(parents=[], subparser=None):
	if subparser:
		compress_args = subparser.add_parser('compress',
			help='Compress datapoints to a DynamicTree',
			conflict_handler='resolve',
			parents=parents
			)
	else:
		compress_args = ArgumentParser(
			description='Compress datapoints to a DynamicTree',
			conflict_handler='resolve',
			parents=parents
			)
	
	compress_args.add_argument(
		'--datapoints', '-X',
		required=True,
		metavar='PATH',
		help='A path to a file of datapoints as .bin'
		)
	
	compress_args.add_argument(
		'--xtype', '-t',
		metavar='TYPE',
		default='float',
		help='The expected data-type of the datapoints'
		)
	
	compress_args.add_argument(
		'--dim', '-d',
		type=int,
		metavar='INT',
		default=3,
		help='The expected dimension of the datapoints'
		)
	
	compress_args.add_argument(
		'--dims', '-D',
		type=int,
		nargs='*',
		metavar='INT',
		default=[],
		help='Dimension per tree layer'
		)
	
	compress_args.add_argument(
		'--qtype', '-q',
		metavar='TYPE',
		default='object',
		help='The quantization type for the datapoints'
		)
	
	compress_args.add_argument(
		'--bits_per_dim', '-B',
		type=int,
		nargs='*',
		metavar='INT',
		default=[16, 16, 16],
		help='The quantization size per dimension'
		)
	
	compress_args.add_argument(
		'--tree_depth', '-T',
		type=int,
		metavar='INT',
		default=48,
		help='The expected dimension of the datapoints'
		)
	
	compress_args.add_argument(
		'--breadth_first', '-b',
		action='store_true',
		help='Flag whether the tree-structure is either breadth first or (default) depth first'
		)
	
	compress_args.add_argument(
		'--payload', '-p',
		action='store_true',
		help='Flag whether or (default) not to separate a payload file'
		)
	
	compress_args.add_argument(
		'--sort_bits', '-P',
		action='store_true',
		help='Flag whether the bits of the datapoints get either sorted by probability or (default) not'
		)
	
	compress_args.add_argument(
		'--reverse', '-r',
		action='store_true',
		help='Flag whether the DynamicTree starts from either heigher or (default) lower bit'
		)
	
	compress_args.set_defaults(
		run=compress
		)
	
	return compress_args


def init_decompress_args(parents=[], subparser=None):
	if subparser:
		decompress_args = subparser.add_parser('decompress',
			help='Decompress a DynamicTree to datapoints',
			conflict_handler='resolve',
			parents=parents
			)
	else:
		decompress_args = ArgumentParser(
			description='Decompress a DynamicTree to datapoints',
			conflict_handler='resolve',
			parents=parents
			)
	
	decompress_args.add_argument(
		'--header_file', '-Y',
		required=True,
		metavar='PATH',
		help='A path to a header file as .hdr.pkl'
		)
	
	decompress_args.set_defaults(
		run=decompress
		)
	
	return decompress_args


def init_kitti_args(parents=[], subparser=None):
	if subparser:
		kitti_args = subparser.add_parser('kitti',
			help='Compress kitti data to a DynamicTree',
			conflict_handler='resolve',
			parents=parents
			)
	else:
		kitti_args = ArgumentParser(
			description='Compress kitti data to a DynamicTree',
			conflict_handler='resolve',
			parents=parents
			)
	
	kitti_args.add_argument(
		'--kittidata', '-X',
		required=True,
		metavar='WILDCARD',
		help='A wildcard to kitti lidar scans'
		)
	
	kitti_args.add_argument(
		'--limit', '-L',
		type=int,
		metavar='INT',
		default=0,
		help='Limit chunk size'
		)
	
	kitti_args.add_argument(
		'--dims', '-D',
		type=int,
		nargs='*',
		metavar='INT',
		default=[],
		help='Dimension per tree layer'
		)
	
	kitti_args.add_argument(
		'--scale', '-S',
		type=float,
		nargs='*',
		default=[200.0, 200.0, 30.0, 1.0],
		metavar='FLOAT',
		help='Scaleing factors for the kitti data'
		)
	
	kitti_args.add_argument(
		'--offset', '-O',
		type=float,
		nargs='*',
		default=[-100.0, -100.0, -25.0, 0],
		metavar='FLOAT',
		help='Offsets for the kitti data'
		)
	
	kitti_args.add_argument(
		'--bits_per_dim', '-B',
		type=int,
		nargs='*',
		default=[16, 16, 8, 8],
		metavar='INT',
		help='Bits per dim for quantization'
		)
	
	kitti_args.add_argument(
		'--qtype', '-q',
		metavar='TYPE',
		default='uint64',
		help='The quantization type for the datapoints'
		)
	
	kitti_args.add_argument(
		'--tree_depth', '-T',
		type=int,
		metavar='INT',
		default=64,
		help='The expected dimension of the datapoints'
		)
	
	kitti_args.add_argument(
		'--breadth_first', '-b',
		action='store_true',
		help='Flag whether the tree-structure is either breadth first or (default) depth first'
		)
	
	kitti_args.add_argument(
		'--payload', '-p',
		action='store_true',
		help='Flag whether or (default) not to separate a payload file'
		)
	
	kitti_args.add_argument(
		'--sort_bits', '-P',
		action='store_true',
		help='Flag whether the bits of the datapoints get either sorted by probability or (default) not'
		)
	
	kitti_args.add_argument(
		'--reverse', '-r',
		action='store_true',
		help='Flag whether the DynamicTree starts from either heigher or (default) lower bit'
		)
	
	kitti_args.set_defaults(
		run=kitti
		)


def save_header(header_file, **kwargs):
	with open(header_file, 'wb') as fid:
		pickle.dump(kwargs, fid),
	return header_file, kwargs


def load_header(header_file, **kwargs):
	with open(header_file, 'rb') as fid:
		header = pickle.load(fid)
	return Prototype(**header)


def load_datapoints(datapoints, xtype=np.float, dim=3, **kwargs):
	X = np.fromfile(datapoints, dtype=xtype)
	X = X[:(len(X)//dim)*dim].reshape(-1,dim)
	X = np.unique(X, axis=0)
	return X
	

def compress(datapoints,
	dims=[],
	bits_per_dim=[16,16,16],
	tree_depth=None,
	output=None,
	breadth_first=False,
	sort_bits=False,
	reverse=False,
	xtype=np.float,
	qtype=object,
	**kwargs
	):
	"""
	"""
	if output is None:
		output = datapoints
	if output:
		output = path.splitext(output)[0]

	X = load_datapoints(datapoints, xtype=xtype, **kwargs)
	X, offset, scale = bitops.serialize(X, bits_per_dim, qtype=qtype)
	if sort_bits:
		X, permute = bitops.sort_bits(X, reverse)
		permute = permute.tolist()
	elif args.reverse:
		X = reverse_bits(X)
		permute = True
	else:
		permute = False

	log("\nData:", X.shape)
	log(X)
	log("\n---Encoding---\n")
	
	flags, payload = dynamictree.encode(X,
		dims=dims,
		tree_depth=tree_depth,
		output=output,
		breadth_first=breadth_first,
		**kwargs
		)
	
	header_file, header = save_header(
		output + '.hdr.pkl',
		dims=dims,
		flags = path.basename(flags.name),
		payload = path.basename(payload.name) if payload else False,
		num_points = len(X),
		breadth_first = breadth_first,
		offset = offset.tolist(),
		scale = scale.tolist(),
		permute = permute,
		bits_per_dim=bits_per_dim,
		xtype = xtype,
		qtype = qtype,
		)
	
	log("\n")
	log("Header saved to:", header_file)
	log("Flags saved to:", flags.name)
	log("Payload saved to:", payload.name)
	return flags, payload, header


def decompress(header_file, output=None, **kwargs):
	"""
	"""
	if output is None:
		output = header_file
	if output:
		output = path.splitext(path.splitext(output)[0])[0] + '.bin'
	
	header = load_header(header_file)
	log("\n---Header---")
	log("\n".join(["{}: {}".format(k,v) for k,v in header.__dict__.items()]))
	log("---Header---")
	
	header.flags = path.join(path.dirname(header_file), header.flags)
	header.payload = path.join(path.dirname(header_file), header.payload) if header.payload else None
	
	flags = BitBuffer(header.flags, 'rb')
	log("\n---Decoding---\n")
	X = dynamictree.decode(flags, **header.__dict__)
	
	if header.permute is True:
		X = reverse_bits(X)
	elif header.permute:
		X = bitops.permute_bits(X, header.permute)
	
	X = bitops.deserialize(X, header.bits_per_dim, header.qtype)
	X = bitops.realization(X, header.offset, header.scale)
	X.tofile(output)
	log("\nData:", X.shape)
	log(X)
	log("Datapoints saved to:", output)
	return X


def merge_frames(frames,
	bits_per_dim=[16, 16, 8, 8],
	offset=[-100.0, -100.0, -25.0, 0],
	scale=[200.0, 200.0, 30.0, 1.0],
	limit=0,
	qtype=np.uint64
	):
	"""
	"""
	scale = (1<<np.array(bits_per_dim) - 1).astype(float) / scale
	for i, X in enumerate(frames):
		X = bitops.serialize(X, bits_per_dim, qtype=qtype, offset=offset, scale=scale)[0]
		X |= i << np.sum(bits_per_dim)
		yield X
		if limit and i >= limit-1:
			break


def kitti(kittidata,
	dims=[],
	bits_per_dim=[16, 16, 8, 8],
	offset=[-100.0, -100.0, -25.0, 0],
	scale=[200.0, 200.0, 30.0, 1.0],
	limit=0,
	qtype=np.uint64,
	output=None,
	tree_depth=None,
	breadth_first=False,
	sort_bits=False,
	reverse=False,
	**kwargs
	):
	"""
	"""
	from pykitti.utils import yield_velo_scans
	files = ifile(kittidata)    
	frames = yield_velo_scans(files)
	
	if output is None:
		output = path.dirname(kittidata)
	if output:
		output = path.splitext(output)[0]
	
	i = 0
	while True:
		output_i = '{}_{:0>4}'.format(output, i)
		X = np.hstack([X for X in merge_frames(frames, bits_per_dim, offset, scale, limit, qtype)])
		X = np.unique(X, axis=0)
		if len(X) == 0:
			return
		else:
			i += 1
		
		if sort_bits:
			X, permute = bitops.sort_bits(X, reverse)
			permute = permute.tolist()
		elif args.reverse:
			X = reverse_bits(X)
			permute = True
		else:
			permute = False

		log("\nChunk No.", i)
		log("Data:", X.shape)
		log("Range:", X.max(axis=0))
		log(X)
		log("\n---Encoding---\n")
		
		flags, payload = dynamictree.encode(X,
			dims=dims,
			tree_depth=tree_depth,
			output=output_i,
			breadth_first=breadth_first,
			**kwargs
			)
		
		header_file, header = save_header(
			output_i + '.hdr.pkl',
			dims = dims,
			flags = path.basename(flags.name),
			payload = path.basename(payload.name) if payload is not None else False,
			num_points = len(X),
			breadth_first = breadth_first,
			offset = offset,
			scale = scale,
			permute = permute,
			bits_per_dim=bits_per_dim,
			xtype = float,
			qtype = qtype
			)
		
		log("\n")
		log("---Header---")
		log("\n".join(["{}: {}".format(k,v) for k,v in header.items()]))
		log("---Header---")
		
		log("\nHeader saved to:", header_file)
		log("Flags saved to:", flags.name)
		log("Payload saved to:", payload.name)
	pass


def main(args, unparsed):
	log.verbose = args.verbose
	args.run(**args.__dict__)


if __name__ == '__main__':
	main_args = init_main_args()
	subparser = main_args.add_subparsers(help='Application Modes:')
	init_compress_args([main_args], subparser)
	init_decompress_args([main_args], subparser)
	init_kitti_args([main_args], subparser)
	main(*main_args.parse_known_args())
	