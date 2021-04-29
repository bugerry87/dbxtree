#!/usr/bin/env python3

## Build in
from argparse import ArgumentParser
import os.path as path
import pickle

## Installed
import numpy as np

## Local
import mhdm.bitops as bitops
from mhdm.utils import Prototype, log, ifile
from mhdm.bitops import BitBuffer
from mhdm.lidar import save


def init_main_args(parents=[]):
	"""
	Initialize an ArgumentParser for this module.
	
	Args:
		parents: A list of ArgumentParsers of other scripts, if there are any.
		
	Returns:
		main_args: The ArgumentParsers.
	"""
	main_args = ArgumentParser(
		description="NbitTree",
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
			help='Encode datapoints to a NbitTree',
			conflict_handler='resolve',
			parents=parents
			)
	else:
		encode_args = ArgumentParser(
			description='Encode datapoints to a NbitTree',
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
		default=None,
		metavar='FLOAT',
		help='Scaling factors for the kitti data'
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
		help='Dimension per tree layer in a range from 0 to 6 (default=0)'
			+ '\nNote: Counter-Trees can be implied by --dims 0'
			+ '\nWarning: dim=0 cannot be folloed after higher dimensions'
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
		'--payload', '-p',
		action='store_true',
		help='Flag whether to separate a payload file or (default) not'
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
			help='Decode a NbitTree to datapoints',
			conflict_handler='resolve',
			parents=parents
			)
	else:
		decode_args = ArgumentParser(
			description='Decode a NbitTree to datapoints',
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


def init_evaluation_args(parents=[], subparser=None):
	if subparser:
		evaluation_args = subparser.add_parser('evaluate',
			help='Evaluate a NbitTrees',
			conflict_handler='resolve',
			parents=parents
			)
	else:
		evaluation_args = ArgumentParser(
			description='Evaluate a NbitTrees',
			conflict_handler='resolve',
			parents=parents
			)
	
	evaluation_args.add_argument(
		'--header_files', '-Y',
		required=True,
		metavar='WILDCARD',
		help='A wildcard to header files as .hdr.pkl'
		)
	
	evaluation_args.set_defaults(
		run=evaluate
		)
	
	return evaluation_args


def save_header(header_file, **kwargs):
	with open(header_file, 'wb') as fid:
		pickle.dump(kwargs, fid)
	return header_file, Prototype(**kwargs)


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
	
	if len(files) == 1 and limit != 1:
		X = np.fromfile(files[0], dtype=xtype)
		X = X[:(len(X)//dim)*dim].reshape(-1,dim)
		yield np.vstack([x for x in split(X)]), files
	else:
		files = iter(files)
		while True:
			processed = []
			A = [merge(f, i) for i, f in zip(range(limit), files)]
			if A:
				yield np.vstack(A), processed
			else:
				break


def yield_dims(dims, word_length):
	if len(dims):
		prev = 0
		for dim in dims:
			if dim > 6:
				raise ValueError("Tree dimension greater than 6 is not allowed!")
			if prev > 0 and dim <= 0:
				raise ValueError("Tree dimension of '0' cannot be followed after higher dimensions!")
			prev = dim
			if word_length > 0:
				word_length -= max(dim, 1)
				yield dim
		while word_length > 0:
			word_length -= max(dim, 1)
			yield dim
	else:
		while word_length > 0:
			word_length -= 1
			yield 0


def encode(files,
	dims=[],
	bits_per_dim=[16,16,16],
	offset=None,
	scale=None,
	bits_for_chunk_id=8,
	output='',
	payload=False,
	sort_bits=False,
	absolute=False,
	reverse=False,
	xtype=np.float32,
	qtype=object,
	limit=1,
	**kwargs
	):
	"""
	"""
	files = [f for f in ifile(files)]
	output = path.splitext(output)[0]
	tree = BitBuffer()
	payload = payload and BitBuffer()
	inp_dim = len(bits_per_dim)
	if limit > 1:
		bits_per_dim = bits_per_dim + [bits_for_chunk_id]
	word_length = sum(bits_per_dim)
	bpp_avg = 0
	bpp_min = 1<<31
	bpp_max = 0
	
	for PC, processed in yield_merged_data(files, xtype, inp_dim, limit):
		if len(files) == 1:
			output_file = output if output else processed[0]
		elif limit == 1:
			output_file = "{}_{}".format(output, processed[0])
		else:
			output_file = "{}_{}-{}".format(output, processed[0], processed[-1])
		
		X, _offset, _scale = bitops.serialize(PC, bits_per_dim, qtype, offset, scale)
		if sort_bits or absolute:
			X, permute, mask = bitops.sort(X, word_length, reverse, absolute)
			permute = permute.tolist()
		elif reverse:
			X = bitops.reverse(X, word_length)
			permute = True
		else:
			permute = False
		X = np.unique(X)

		if log.verbose:
			log("\nChunk:", output_file)
			log("Data:", X.shape)
			for x in X[::len(X)//10]:
				log("{:0>16}".format(hex(x)[2:]))
			log("...")
			log("\n---Encoding---\n")

		dim_seq = [dim for dim in yield_dims(dims, word_length)]
		layers = bitops.tokenize(X, dim_seq)
		mask = np.ones(len(X), bool)
		tail = np.full(len(X), word_length)
		tree.open(output_file + '.flg.bin', 'wb')
		payload and payload.open(output_file + '.flg.bin', 'wb')
		for i, (X0, X1, dim) in enumerate(zip(layers[:-1], layers[1:], dim_seq)):
			uids, idx, counts = np.unique(X0[mask], return_inverse=True, return_counts=True)
			flags, hist = bitops.encode(X1[mask], idx, max(dim,1))
			if payload:
				mask[mask] = counts > 1
				tail[mask] -= dim
			for flag, val, count in zip(flags, hist[:,reverse], counts):
				if dim:
					tree.write(flag, 1<<dim, soft_flush=True)
				else:
					bits = max(int(count).bit_length(), 1)
					tree.write(val, bits, soft_flush=True)
			log(".", end='', flush=True)
		
		if payload:
			for x, bits in zip(X, tail):
				payload.write(x, bits, soft_flush=True)
		
		log("\r\n")
		header_file, header = save_header(
			output_file + '.hdr.pkl',
			dims=dims,
			flags = path.basename(tree.name),
			payload = path.basename(payload.name) if payload else False,
			inp_points = len(PC),
			out_points = len(X),
			offset = _offset.tolist(),
			scale = _scale.tolist(),
			permute = permute,
			bits_per_dim=bits_per_dim,
			xtype = xtype,
			qtype = qtype
			)
		log("---Header---")
		log("\n".join(["{}: {}".format(k,v) for k,v in header.__dict__.items()]))		
		log("\n")
		log("Header saved to:", header_file)
		
		log("Flags saved to:", tree.name)
		if payload:
			log("Payload saved to:", payload.name)
		
		if log.verbose:
			bpp = ((payload and len(payload)) + len(tree)) / len(X)
			log("Bit per Points (bpp):", bpp)
			bpp_min = min(bpp_min, bpp)
			bpp_max = max(bpp_max, bpp)
			bpp_avg += bpp
	
	tree.close()
	payload and payload.close()
	if log.verbose:
		log("\nSummary:")
		log("bpp avg:", bpp_avg/len(files))
		log("bpp min:", bpp_min)
		log("bpp max:", bpp_max)
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
	
	flags = BitBuffer(path.join(path.dirname(header_file), header.flags), 'rb')
	payload = header.payload and BitBuffer(path.join(path.dirname(header_file), header.payload), 'rb')
	word_length = sum(header.bits_per_dim)
	dim_seq = yield_dims(header.dims, word_length)
	counts = [header.num_points]
	
	log("\n---Decoding---\n")

	X = np.zeros([1], dtype=header.qtype)
	tails = np.full(1, word_length) if payload else None
	for dim in dim_seq:
		if dim:
			nodes = np.array([flags.read(dim) for i in range(len(X))])
		else:
			nodes = np.array([flags.read(int(c).bit_length()) for c in counts])
		X, counts, tails = bitops.decode(nodes, dim, X, tails)
	
	if payload:
		payload = [payload.read(bits) for bits in tails]
		X = X << tails | payload
	
	if header.permute is True:
		X = bitops.reverse(X, word_length)
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


def evaluate(header_files, **kwargs):
	import matplotlib.pyplot as plt

	bpp_all = dict()
	bpp_avg = dict()
	bpp_min = dict()
	bpp_max = dict()
	files = [f for f in ifile(header_files)]
	for f in files:
		header = load_header(f)
		log("\n---Header---")
		log("\n".join(["{}: {}".format(k,v) for k,v in header.__dict__.items()]))

		label = ",".join([str(1<<d) for d in header.dims]) + "bitTree"
		flags = path.getsize(path.join(path.dirname(f), header.flags))
		payload = header.payload and path.getsize(path.join(path.dirname(f), header.payload))

		bpp = (flags + payload) * 8 / header.inp_points
		bpp_all[label] = bpp_all[label] + [bpp] if label in bpp_all else [bpp]
	
	for key in bpp_all.keys():
		bpp_avg[key] = np.mean(bpp_all[key])
		bpp_min[key] = min(bpp_all[key])
		bpp_max[key] = max(bpp_all[key])
	
	colLabels = ['16bitTree', '8bitTree', '4bitTree', '2bitTree']
	cell_text = [['{:2.2f}'.format(d[k]) for k in colLabels] for d in [bpp_max, bpp_avg, bpp_min]]
	rowLabels = ['bpp max', 'bpp avg', 'bpp min']
	

	plt.boxplot([bpp_all[k] for k in colLabels])
	plt.xticks([])
	plt.subplots_adjust(left=0.2, bottom=0.2)
	plt.table(cellText=cell_text, cellLoc='center', rowLabels=rowLabels, colLabels=colLabels, loc='bottom')
	plt.title('Bits per input Points (bpp)')
	plt.show()
	pass


def main(args, unparsed):
	log.verbose = args.verbose
	args.run(**args.__dict__)


if __name__ == '__main__':
	main_args = init_main_args()
	subparser = main_args.add_subparsers(help='Application Modes:')
	init_encode_args([main_args], subparser)
	init_decode_args([main_args], subparser)
	init_evaluation_args([main_args], subparser)
	main(*main_args.parse_known_args())