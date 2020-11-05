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
		default='',
		help='A filename for the output data'
		)
	
	main_args.set_defaults(
		run=lambda **kwargs: main_args.print_help()
		)
	
	return main_args


def init_encode_args(parents=[], subparser=None):
	if subparser:
		encode_args = subparser.add_parser('encode',
			help='Encode datapoints to a DynamicTree',
			conflict_handler='resolve',
			parents=parents
			)
	else:
		encode_args = ArgumentParser(
			description='Encode datapoints to a DynamicTree',
			conflict_handler='resolve',
			parents=parents
			)
	
	encode_args.add_argument(
		'--datapoints', '-X',
		nargs='+',
		metavar='WILDCARD',
		help='One or more wildcards to files of datapoints as .bin'
		)
	
	encode_args.add_argument(
		'--limit', '-L',
		type=int,
		metavar='INT',
		default=1,
		help='Limit chunk size (default=1)'
		)
	
	encode_args.add_argument(
		'--scale', '-S',
		type=float,
		nargs='*',
		default=[],
		metavar='FLOAT',
		help='Scaling factors for the kitti data'
		)
	
	encode_args.add_argument(
		'--offset', '-O',
		type=float,
		nargs='*',
		default=[],
		metavar='FLOAT',
		help='Offsets for the kitti data'
		)
	
	encode_args.add_argument(
		'--xtype', '-t',
		metavar='TYPE',
		default='float32',
		help='The expected data-type of the datapoints (datault=float32)'
		)
	
	encode_args.add_argument(
		'--dims', '-D',
		type=int,
		nargs='*',
		metavar='INT',
		default=[],
		help='Dimension per tree layer'
		)
	
	encode_args.add_argument(
		'--qtype', '-q',
		metavar='TYPE',
		default='uint64',
		help='The quantization type for the datapoints (dafault=uint64)'
		)
	
	encode_args.add_argument(
		'--bits_per_dim', '-B',
		type=int,
		nargs='*',
		metavar='INT',
		default=[16, 16, 16],
		help='The quantization size per dimension'
		)
	
	encode_args.add_argument(
		'--breadth_first', '-b',
		action='store_true',
		help='Flag whether the tree-structure is either breadth first or (default) depth first'
		)
	
	encode_args.add_argument(
		'--payload', '-p',
		action='store_true',
		help='Flag whether or (default) not to separate a payload file'
		)
	
	encode_args.add_argument(
		'--sort_bits', '-P',
		action='store_true',
		help='Flag whether the bits of the datapoints get either sorted by probability or (default) not'
		)
	
	encode_args.add_argument(
		'--reverse', '-r',
		action='store_true',
		help='Flag whether the DynamicTree starts from either heigher or (default) lower bit'
		)
	
	encode_args.set_defaults(
		run=encode
		)
	
	return encode_args


def init_decode_args(parents=[], subparser=None):
	if subparser:
		decode_args = subparser.add_parser('decode',
			help='Decode a DynamicTree to datapoints',
			conflict_handler='resolve',
			parents=parents
			)
	else:
		decode_args = ArgumentParser(
			description='Decode a DynamicTree to datapoints',
			conflict_handler='resolve',
			parents=parents
			)
	
	decode_args.add_argument(
		'--header_file', '-Y',
		required=True,
		metavar='PATH',
		help='A path to a header file as .hdr.pkl'
		)
	
	decode_args.set_defaults(
		run=decode
		)
	
	return decode_args


def init_kitti_args(parents=[], subparser=None):
	if subparser:
		kitti_args = subparser.add_parser('kitti',
			help='Encode kitti data to a DynamicTree',
			conflict_handler='resolve',
			parents=parents
			)
	else:
		kitti_args = ArgumentParser(
			description='Encode kitti data to a DynamicTree',
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
		default=[200.0, 200.0, 200.0, 1.0, 1.0],
		metavar='FLOAT',
		help='Scaling factors for the kitti data'
		)
	
	kitti_args.add_argument(
		'--offset', '-O',
		type=float,
		nargs='*',
		default=[100.0, 100.0, 100.0, 0, 0],
		metavar='FLOAT',
		help='Offsets for the kitti data'
		)
	
	kitti_args.add_argument(
		'--bits_per_dim', '-B',
		type=int,
		nargs='*',
		default=[16, 16, 16, 8, 8],
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


def load_datapoints(files, xtype=np.float32, dim=3, limit=0, **kwargs):
	"""
	"""
	processed = []
	def merge(file, i):
		X = np.fromfile(file, dtype=xtype)
		X = X[:(len(X)//dim)*dim].reshape(-1,dim)
		file = path.basename(file)
		file = path.splitext(file)[0]
		processed.append(file)
		return np.hstack((X, np.full((len(X),1), i, dtype=xtype)))
	return np.vstack([merge(f, i) for f, i in zip(files, range(limit))]), processed
	

def encode(datapoints,
	dims=[],
	bits_per_dim=[16,16,16],
	output='',
	breadth_first=False,
	sort_bits=False,
	reverse=False,
	xtype=np.float32,
	qtype=object,
	limit=1,
	**kwargs
	):
	"""
	"""
	output = path.splitext(output)[0]
	files = ifile(datapoints, sort=True)
	nfiles = len(files)
	files iter(files)
	dim = len(bits_per_dim)
	tree_depth = int(np.sum(bits_per_dim))
	
	while files:
		X, processed = load_datapoints(files, xtype, dim, limit)
		X, offset, scale = bitops.serialize(X, bits_per_dim, qtype=qtype)
		if sort_bits:
			X, permute = bitops.sort(X, tree_depth, reverse, True)
			permute = permute.tolist()
		elif args.reverse:
			X = reverse_bits(X)
			permute = True
		else:
			permute = False
		X = np.unique(X)
		
		if nfiles == 1:
			output_file = output if output else processed[0]
		elif limit == 1:
			output_file = "{}_{}".format(output, processed)
		else:
			output_file = "{}_{}-{}".format(output, processed[0], processed[-1])

		if log.verbose:
			log("\nChunk:",output_file )
			log("Data:", X.shape)
			for x in X[::len(X)//10]:
				log("{:0>16}".format(hex(x)[2:]))
			log("...")
			log("\n---Encoding---\n")
		
		flags, payload = dynamictree.encode(X,
			dims=dims,
			tree_depth=tree_depth,
			output=output_file,
			breadth_first=breadth_first,
			**kwargs
			)
		
		header_file, header = save_header(
			output_file + '.hdr.pkl',
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
		log("---Header---")
		log("\n".join(["{}: {}".format(k,v) for k,v in header.items()]))
		
		log("\n")
		log("Header saved to:", header_file)
		log("Flags saved to:", flags.name)
		if payload:
			log("Payload saved to:", payload.name)
	pass


def decode(header_file, output=None, **kwargs):
	"""
	"""
	if output is None:
		output = header_file
	if output:
		output = path.splitext(path.splitext(output)[0])[0] + '.bin'
	
	header = load_header(header_file)
	log("\n---Header---")
	log("\n".join(["{}: {}".format(k,v) for k,v in header.__dict__.items()]))
	
	header.flags = path.join(path.dirname(header_file), header.flags)
	header.payload = path.join(path.dirname(header_file), header.payload) if header.payload else None
	header.scale = ((1<<np.array(header.bits_per_dim)) - 1).astype(float) / header.scale
	
	flags = BitBuffer(header.flags, 'rb')
	log("\n---Decoding---\n")
	X = dynamictree.decode(flags, **header.__dict__)
	
	if header.permute is True:
		X = bitops.reverse(X)
	elif header.permute:
		X = bitops.permute(X, header.permute)
	
	X = bitops.deserialize(X, header.bits_per_dim, header.qtype)
	X = bitops.realization(X, header.offset, header.scale)
	log("\nData:", X.shape)
	log(np.round(X,2))
	log("Datapoints saved to:", output)
	return X


def merge_frames(frames,
	bits_per_dim=[16, 16, 16],
	offset=[],
	scale=[],
	limit=0,
	qtype=np.uint64
	):
	"""
	"""
	scale = ((1<<np.array(bits_per_dim)) - 1).astype(float) / scale
	for i, X in enumerate(frames):
		X = bitops.serialize(X, bits_per_dim, qtype=qtype, offset=offset, scale=scale)[0]
		X |= 0x0 << int(np.sum(bits_per_dim))
		yield X
		if limit and i >= limit-1:
			break


def kitti(kittidata,
	dims=[],
	bits_per_dim=[16, 16, 16, 8, 8],
	offset=[100.0, 100.0, 100.0, 0, 0],
	scale=[200.0, 200.0, 200.0, 1.0, 1.0],
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
	files = ifile(kittidata)
	frames = (np.fromfile(f, dtype=np.float32).reshape(-1,4) for f in files)
	
	if output is None:
		output = path.dirname(kittidata)
	if output:
		output = path.splitext(output)[0]
	
	i = 0
	while True:
		output_i = '{}_{:0>4}'.format(output, i)
		X = [X for X in merge_frames(frames, bits_per_dim[:-1], offset[:-1], scale[:-1], limit, qtype)]
		if len(X) == 0:
			return
		else:
			i += 1
		X = np.hstack(X)
		X = np.unique(X, axis=0)
		
		if sort_bits:
			X, permute = bitops.sort(X, reverse, True)
			permute = permute.tolist()
		elif reverse:
			X = bitops.reverse(X)
			permute = True
		else:
			permute = False

		if log.verbose:
			log("\nChunk No.", i)
			log("Data:", X.shape)
			X.sort()
			for x in X[::len(X)//10]:
				log("{:0>16}".format(hex(x)[2:]))
			log("...")
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
			payload = path.basename(payload.name) if payload else False,
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
		
		log("\nHeader saved to:", header_file)
		log("Flags saved to:", flags.name)
		if payload:
			log("Payload saved to:", payload.name)
	pass


def main(args, unparsed):
	log.verbose = args.verbose
	args.run(**args.__dict__)


if __name__ == '__main__':
	main_args = init_main_args()
	subparser = main_args.add_subparsers(help='Application Modes:')
	init_encode_args([main_args], subparser)
	init_decode_args([main_args], subparser)
	init_kitti_args([main_args], subparser)
	main(*main_args.parse_known_args())
	
