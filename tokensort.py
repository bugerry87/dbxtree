#!/usr/bin/env python3

## Build in
from argparse import ArgumentParser
from collections import deque
import os.path as path
import pickle

## Installed
import numpy as np

## Local
import mhdm.tokensort as tokensort
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
		description="TokenSort",
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
			help='Encode datapoints to TokenSort',
			conflict_handler='resolve',
			parents=parents
			)
	else:
		encode_args = ArgumentParser(
			description='Encode datapoints to TokenSort',
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
		'--bits_per_dim', '-B',
		type=int,
		nargs='*',
		metavar='INT',
		default=[16, 16, 16],
		help='The quantization size per dimension'
		)
	
	encode_args.add_argument(
		'--scale', '-S',
		type=float,
		nargs='*',
		default=None,
		metavar='FLOAT',
		help='Scaleing factors for the kitti data'
		)
	
	encode_args.add_argument(
		'--offset', '-O',
		type=float,
		nargs='*',
		default=None,
		metavar='FLOAT',
		help='Offsets for the kitti data'
		)
	
	encode_args.add_argument(
		'--qtype', '-q',
		metavar='TYPE',
		default='uint64',
		help='The quantization type for the datapoints'
		)
	
	encode_args.add_argument(
		'--sort', '-s',
		action='store_true',
		help='Flag whether to sort the datapoints or (default) not'
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
	
	encode_args.add_argument(
		'--iterations', '-i',
		type=int,
		metavar='INT',
		default=0,
		help='The number of additional iterations'
		)
	
	return encode_args


def init_decode_args(parents=[], subparser=None):
	if subparser:
		decode_args = subparser.add_parser('decode',
			help='Decode TokenSort to datapoints',
			conflict_handler='resolve',
			parents=parents
			)
	else:
		decode_args = ArgumentParser(
			description='Decode TokenSort to datapoints',
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
		'--chunk', '-C',
		type=int,
		metavar='INT',
		default=1,
		help='Chunk size'
		)
	
	kitti_args.add_argument(
		'--scale', '-S',
		type=float,
		nargs='*',
		default=None,
		metavar='FLOAT',
		help='Scaleing factors for the kitti data'
		)
	
	kitti_args.add_argument(
		'--offset', '-O',
		type=float,
		nargs='*',
		default=None,
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
		'--sort', '-s',
		action='store_true',
		help='Flag whether to sort the datapoints or (default) not'
		)
	
	kitti_args.add_argument(
		'--sort_bits', '-p',
		action='store_true',
		help='Flag whether the bits of the datapoints get either sorted by probability or (default) not'
		)
	
	kitti_args.add_argument(
		'--reverse', '-r',
		action='store_true',
		help='Flag whether the TokenTree starts from either heigher or (default) lower bit'
		)
	
	kitti_args.add_argument(
		'--iterations', '-i',
		type=int,
		metavar='INT',
		default=0,
		help='The number of additional iterations'
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


def load_datapoints(datapoints, xtype=float, dim=3, **kwargs):
	X = np.fromfile(datapoints, dtype=xtype)
	X = X[:(len(X)//dim)*dim].reshape(-1,dim)
	return X
	

def encode(datapoints, bits_per_dim,
	output=None,
	xtype=float,
	qtype=object,
	**kwargs
	):
	"""
	"""
	if output is None:
		output = datapoints
	if output:
		output = path.splitext(output)[0]
	payload_file = output + '.pyl.bin'

	X = load_datapoints(datapoints, xtype, len(bits_per_dim))
	log("\nData:", X.shape)
	log(X)
	log("\n---Encoding---\n")
	X, offset, scale, permutes, zero_padding = tokensort.encode(X, bits_per_dim, qtype, **kwargs)
	X.tofile(payload_file)
	log(X)
	
	header_file, header = save_header(
		output + '.hdr.pkl',
		payload_file = payload_file,
		bits_per_dim = bits_per_dim,
		permutes = permutes,
		zero_padding = zero_padding,
		offset = offset,
		scale = scale,
		qtype = qtype,
		xtype = xtype,
		)
	
	log("\n")
	log("Header saved to:", header_file)
	log("Payload saved to:", payload_file)
	return X, header


def decode(header_file, output=None, **kwargs):
	"""
	"""
	if output is None:
		output = header_file
	if output:
		output = path.splitext(output)[0]
	
	header = load_header(header_file)
	header.payload = path.join(path.dirname(header_file), header.payload)
	
	Y = np.fromfile(header.payload, dtype=header.qtype)
	log("\nTokens:", Y.shape)
	log(Y)
	log("\n---Decoding---\n")
	X = tokensort.decode(Y, **header.__dict__)
	X.tofile(output + '.bin')
	
	log("\nData:", X.shape)
	log(X)
	log("Datapoints saved to:", output)
	return X


def tag_frames(frames, chunk=1):
	for i, X in enumerate(frames):
		if i < chunk:
			yield np.hstack((X, np.full((len(X),1), i)))


def kitti(kittidata, bits_per_dim,
	chunk=1,
	output=None,
	qtype=np.uint64,
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
		output_file = '{}_{:0>4}'.format(output, i)
		payload_file = output_file + '.pyl.bin'
		X = [X for X in tag_frames(frames, chunk)]
		if len(X) == 0:
			return
		else:
			i += 1
		X = np.vstack(X)
		X = np.unique(X, axis=0)

		log("\nChunk No.", i)
		log("Data:", X.shape)
		log(X)
		log("\n---Encoding---\n")
		
		X, offset, scale, permutes, zero_padding = tokensort.encode(X, bits_per_dim, qtype=qtype, **kwargs)
		X.tofile(payload_file)
		
		header_file, header = save_header(
			output + '.hdr.pkl',
			payload_file = payload_file,
			bits_per_dim = bits_per_dim,
			permutes = permutes,
			zero_padding = zero_padding,
			offset = offset,
			scale = scale,
			qtype = qtype,
			xtype = float,
		)
		
		log("\n")
		log("Header saved to:", header_file)
		log("Payload saved to:", payload_file)
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
	