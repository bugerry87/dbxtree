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
from mhdm.utils import Prototype, log
from mhdm.bitops import BitBuffer


def init_argparsers(parents=[]):
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
	
	subparsers = main_args.add_subparsers(help='Application Mode')
	compress_args = subparsers.add_parser('compress',
		help='Compress datapoints to a TokenTree',
		conflict_handler='resolve',
		parents=[main_args]
		)
	decompress_args = subparsers.add_parser('decompress',
		help='Decompress a TokenTree to datapoints',
		conflict_handler='resolve',
		parents=[main_args]
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
		'--qtype', '-q',
		metavar='TYPE',
		default='uint16',
		help='The quantization type for the datapoints'
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
		'--sort_bits', '-B',
		action='store_true',
		help='Flag whether the bits of the datapoints get either sorted by probability or (default) not'
		)
	
	compress_args.add_argument(
		'--reverse', '-r',
		action='store_true',
		help='Flag whether the TokenTree starts from either heigher or (default) lower bit'
		)
	
	compress_args.set_defaults(
		run=compress
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
	
	return main_args


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
	

def compress(datapoints,
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
	
	flags, payload = tokentree.encode(X,
		output=output,
		breadth_first=breadth_first,
		**kwargs
		)
	
	header_file, header = save_header(
		output + '.hdr.pkl',
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


def decompress(header_file, output=None, **kwargs):
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


def main(args):
	log.verbose = args.verbose
	args.run(**args.__dict__)


if __name__ == '__main__':
	main_parser = init_argparsers()
	args = main_parser.parse_args()
	main(args)
		