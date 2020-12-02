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
from mhdm.lidar import save
from mhdm.probtree import ProbTree


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
		'--files', '-X',
		nargs='+',
		metavar='WILDCARD',
		help='One or more wildcards to files of datapoints as .bin'
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
		'--limit', '-L',
		type=int,
		metavar='INT',
		default=1,
		help='Limit chunk size (default=1)'
		)
	
	encode_args.add_argument(
		'--bits_for_chunk_id', '-s',
		type=int,
		metavar='INT',
		default=8,
		help='Bit reservation for scan ID (default=8)'
		)
	
	encode_args.add_argument(
		'--breadth_first', '-b',
		action='store_true',
		help='Flag whether the tree-structure is either breadth first or (default) depth first'
		)
	
	encode_args.add_argument(
		'--flags', '-f',
		action='store_false',
		help='Flag whether to store branching-flags (default) or not'
		)
	
	encode_args.add_argument(
		'--payload', '-p',
		action='store_true',
		help='Flag whether to separate a payload file or (default) not'
		)
	
	encode_args.add_argument(
		'--model', '-M',
		metavar='PATH',
		default=None,
		help='A filename for the model data'
		)
	
	encode_args.add_argument(
		'--sort_bits', '-P',
		action='store_true',
		help="Flag whether the bits of the datapoints get either sorted by the probability to be '1' or (default) not"
		)
	
	encode_args.add_argument(
		'--absolute', '-A',
		action='store_true',
		help='Flag whether the bits of the datapoints get either sorted by absolute probability or (default) not'
		)
	
	encode_args.add_argument(
		'--reverse', '-r',
		action='store_true',
		help='Flag whether to start from either heigher or (default) lower bit'
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
	
	decode_args.add_argument(
		'--formats', '-f',
		metavar='EXT',
		nargs='*',
		choices=('bin', 'npy', 'ply', 'pcd'),
		help='A list of additional output formats'
		)
	
	decode_args.add_argument(
		'--payload', '-p',
		action='store_false',
		help='Flag whether to ignore payload or (default) not'
		)
	
	decode_args.set_defaults(
		run=decode
		)
	
	return decode_args


def save_header(header_file, **kwargs):
	with open(header_file, 'wb') as fid:
		pickle.dump(kwargs, fid)
	return header_file, kwargs


def load_header(header_file, **kwargs):
	with open(header_file, 'rb') as fid:
		header = pickle.load(fid)
	return Prototype(**header)


def yield_merged_data(files, xtype=np.float32, dim=3, limit=1, **kwargs):
	"""
	"""
	def merge(file, i):
		X = np.fromfile(file, dtype=xtype)
		X = X[:(len(X)//dim)*dim].reshape(-1,dim)
		file = path.basename(file)
		file = path.splitext(file)[0]
		processed.append(file)
		if limit == 1:
			return X
		else:
			return np.hstack((X, np.full((len(X),1), i, dtype=xtype)))
	
	def split(X):
		for i, low in enumerate(range(0, len(X), limit)):
			high = min(low+limit, len(X))
			i = np.full((high-low,1), i, dtype=xtype)
			yield np.hstack((X[low:high] - X[low], i))
	
	files = [f for f in ifile(files)]
	if len(files) == 1 and limit != 1:
		X = np.fromfile(files[0], dtype=xtype)
		X = X[:(len(X)//dim)*dim].reshape(-1,dim)
		yield np.vstack([x for x in split(X)]), files
	else:
		files = iter(files)
		while True:
			processed = []
			A = [merge(f, i) for f, i in zip(files, range(limit))]
			if A:
				yield np.vstack(A), processed
			else:
				break
	

def encode(files,
	dims=[],
	bits_per_dim=[16,16,16],
	bits_for_chunk_id=8,
	output='',
	breadth_first=False,
	sort_bits=False,
	absolute=False,
	reverse=False,
	xtype=np.float32,
	qtype=object,
	limit=1,
	model=None,
	**kwargs
	):
	"""
	"""
	output = path.splitext(output)[0]
	dim = len(bits_per_dim)
	if limit > 1:
		bits_per_dim = bits_per_dim + [bits_for_chunk_id]
	tree_depth = sum(bits_per_dim)
	
	if isinstance(model, str):
			model = ProbTree(model, breadth_first)
	
	for X, processed in yield_merged_data(files, xtype, dim, limit):
		X, offset, scale = bitops.serialize(X, bits_per_dim, qtype=qtype)
		if sort_bits or absolute:
			X, permute = bitops.sort(X, tree_depth, reverse, absolute)
			permute = permute.tolist()
		elif reverse:
			X = bitops.reverse(X, tree_depth)
			permute = True
		else:
			permute = False
		X = np.unique(X)
		
		if processed[0] is processed[-1]:
			output_file = output if output else processed[0]
		elif limit == 1:
			output_file = "{}_{}".format(output, processed)
		else:
			output_file = "{}_{}-{}".format(output, processed[0], processed[-1])

		if log.verbose:
			log("\nChunk:", output_file)
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
			model=model,
			**kwargs
			)
		
		log("\n")
		if flags:
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
			log("---Header---")
			log("\n".join(["{}: {}".format(k,v) for k,v in header.items()]))		
			log("\n")
			log("Header saved to:", header_file)
		
		if flags:
			log("Flags saved to:", flags.name)
		if payload:
			log("Payload saved to:", payload.name)
		if model:
			model.save(model.name)
			log("Model saved to:", model.name)
	pass


def decode(header_file, output=None, formats=None, payload=True, **kwargs):
	"""
	"""
	if formats:
		formats = {*formats}
	else:
		formats = set()
	
	if output:
		output, format = path.splitext(output)
		formats.add(format.split('.')[-1])
	else:
		output = path.splitext(header_file)[0]
		output = path.splitext(output)[0]
		if not formats:
			formats.add('bin')
	
	header = load_header(header_file)
	log("\n---Header---")
	log("\n".join(["{}: {}".format(k,v) for k,v in header.__dict__.items()]))
	
	header.flags = path.join(path.dirname(header_file), header.flags)
	if header.payload and payload:
		header.payload = path.join(path.dirname(header_file), header.payload)
	else:
		header.payload = None
	tree_depth = sum(header.bits_per_dim)
	
	flags = BitBuffer(header.flags, 'rb')
	log("\n---Decoding---\n")
	X = dynamictree.decode(flags, tree_depth=tree_depth, **header.__dict__)
	
	if header.permute is True:
		X = bitops.reverse(X, tree_depth)
	elif header.permute:
		X = bitops.permute(X, header.permute)
	
	X = bitops.deserialize(X, header.bits_per_dim, header.qtype)
	X = bitops.realization(X, header.offset, header.scale, header.xtype)
	log("\nData:", X.shape)
	log(np.round(X,2))
	
	for format in formats:
		output_file = save(X, output, format)
		log("Datapoints saved to:", output_file)
	return X


def main(args, unparsed):
	log.verbose = args.verbose
	args.run(**args.__dict__)


if __name__ == '__main__':
	main_args = init_main_args()
	subparser = main_args.add_subparsers(help='Application Modes:')
	init_encode_args([main_args], subparser)
	init_decode_args([main_args], subparser)
	main(*main_args.parse_known_args())
	
