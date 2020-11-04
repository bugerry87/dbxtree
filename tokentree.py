#!/usr/bin/env python3

## Build in
from argparse import ArgumentParser
from collections import deque
import os.path as path
import pickle

## Installed
import numpy as np

## Local
import mhdm.tokentree as tokentree
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
		description="TokenTree",
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


def init_encode_args(parents=[], subparser=None):
	if subparser:
		encode_args = subparser.add_parser('encode',
			help='Encode datapoints to a TokenTree',
			conflict_handler='resolve',
			parents=parents
			)
	else:
		encode_args = ArgumentParser(
			description='Encode datapoints to a TokenTree',
			conflict_handler='resolve',
			parents=parents
			)
	
	encode_args.add_argument(
		'--datapoints', '-X',
		required=True,
		metavar='PATH',
		help='A path to a file of datapoints as .bin'
		)
	
	encode_args.add_argument(
		'--xtype', '-t',
		metavar='TYPE',
		default='float',
		help='The expected data-type of the datapoints'
		)
	
	encode_args.add_argument(
		'--dim', '-d',
		type=int,
		metavar='INT',
		default=3,
		help='The expected dimension of the datapoints'
		)
	
	encode_args.add_argument(
		'--qtype', '-q',
		metavar='TYPE',
		default='uint16',
		help='The quantization type for the datapoints'
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
		help='Flag whether the TokenTree starts from either heigher or (default) lower bit'
		)
	
	encode_args.set_defaults(
		run=encode
		)
	
	return encode_args


def init_decode_args(parents=[], subparser=None):
	if subparser:
		decode_args = subparser.add_parser('decode',
			help='Decode a TokenTree to datapoints',
			conflict_handler='resolve',
			parents=parents
			)
	else:
		decode_args = ArgumentParser(
			description='Decode a TokenTree to datapoints',
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
			help='Encode kitti data to a TokenTree',
			conflict_handler='resolve',
			parents=parents
			)
	else:
		kitti_args = ArgumentParser(
			description='Encode kitti data to a TokenTree',
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
		default=[100.0, 100.0, 25.0, 0],
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
		help='Flag whether the TokenTree starts from either heigher or (default) lower bit'
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


def load_flags(flags, dim=3, **kwargs):
	if dim == 3:
		Y = np.fromfile(flags, dtype=np.uint8)
	elif dim == 4:
		Y = np.fromfile(flags, dtype=np.uint16)
	elif dim == 5:
		Y = np.fromfile(flags, dtype=np.uint32)
	elif dim == 6:
		Y = np.fromfile(flags, dtype=np.uint64)
	else:
		Y = BitBuffer(flags, 'rb', 1<<dim)
	return Y
	

def encode(datapoints,
	output=None,
	breadth_first=False,
	sort_bits=False,
	reverse=False,
	xtype=np.float,
	qtype=np.uint16,
	**kwargs
	):
	"""
	"""
	if output is None:
		output = datapoints
	if output:
		output = path.splitext(output)[0]

	X = load_datapoints(datapoints, xtype=xtype, **kwargs)
	X, offset, scale = bitops.quantization(X, qtype=qtype)
	if sort_bits:
		X, permute = bitops.sort(X, reverse, True)
		permute = permute.tolist()
	elif args.reverse:
		X = reverse_bits(X)
		permute = True
	else:
		permute = False

	log("\nData:", X.shape)
	log(X)
	log("\n---Encoding---\n")
	
	flags, payload = tokentree.encode(X,
		output=output,
		breadth_first=breadth_first,
		**kwargs
		)
	
	header_file, header = save_header(
		output + '.hdr.pkl',
		num_points = len(X),
		flags = path.basename(flags.name),
		payload = path.basename(payload.name) if payload else False,
		breadth_first = breadth_first,
		offset = offset.tolist(),
		scale = scale.tolist(),
		permute = permute,
		xtype = xtype,
		qtype = qtype,
		)
	
	log("\n")
	log("Header saved to:", header_file)
	log("Flags saved to:", flags.name)
	log("Payload saved to:", payload.name)
	return flags, payload, header


def decode(header_file, output=None, **kwargs):
	"""
	"""
	if output is None:
		output = header_file
	if output:
		output = path.splitext(output)[0]
	
	header = load_header(header_file)
	header.flags = path.join(path.dirname(header_file), header.flags)
	header.payload = path.join(path.dirname(header_file), header.payload) if header.payload else None
	
	Y = load_flags(**header.__dict__)
	log("\nFlags:", Y.shape)
	log(Y)
	log("\n---Decoding---\n")
	X = tokentree.decode(Y, **header.__dict__)
	
	if header.permute is True:
		X = reverse_bits(X)
	elif header.permute:
		X = permute_bits(X, header.permute)
	
	X = realization(X, header.offset, header.scale)
	X.tofile(output + '.bin')
	log("\nData:", X.shape)
	log(X)
	log("Datapoints saved to:", output)
	return X


def merge_frames(frames,
	bits_per_dim=[16, 16, 8, 8],
	offset=[100.0, 100.0, 25.0, 0,],
	scale=[200.0, 200.0, 30.0, 1.0],
	limit=0
	):
	"""
	"""
	scale = (1<<np.array(bits_per_dim) - 1).astype(float) / scale
	for i, X in enumerate(frames):
		X = bitops.serialize(X, bits_per_dim, qtype=np.uint64, offset=offset, scale=scale)[0]
		X = np.ndarray((len(X), 4), dtype=np.uint16, buffer=X)
		X[:,-1] = i
		yield X
		if limit and i >= limit-1:
			break


def kitti(kittidata,
	bits_per_dim=[16, 16, 8, 8],
	offset=[100.0, 100.0, 25.0, 0],
	scale=[200.0, 200.0, 30.0, 1.0],
	limit=0,
	output=None,
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
		X = [X for X in merge_frames(frames, bits_per_dim, offset, scale, limit)]
		if len(X) == 0:
			return
		else:
			i += 1
		X = np.vstack(X)
		X = np.unique(X, axis=0)
		
		if sort_bits:
			X, permute = bitops.sort(X, reverse, True)
			permute = permute.tolist()
		elif args.reverse:
			X = bitops.reverse(X)
			permute = True
		else:
			permute = False

		log("\nChunk No.", i)
		log("Data:", X.shape)
		log("Range:", X.max(axis=0))
		log(X)
		log("\n---Encoding---\n")
		
		flags, payload = tokentree.encode(X,
			output=output_i,
			breadth_first=breadth_first,
			**kwargs
			)
		
		header_file, header = save_header(
			output_i + '.hdr.pkl',
			num_points = len(X),
			flags = path.basename(flags.name),
			payload = path.basename(payload.name) if payload else False,
			breadth_first = breadth_first,
			offset = offset,
			scale = scale,
			permute = permute,
			bits_per_dim=bits_per_dim
			)
		
		log("\n")
		log("Header saved to:", header_file)
		log("Flags saved to:", flags.name)
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
	
