#!/usr/bin/env python3

## Build in
from argparse import ArgumentParser
from collections import deque

## Local
from utils import log
from bitops import BitBuffer, quantization


def encode(X, output=None, breadth_first=False, payload=False, **kwargs):
	token_dim = X.shape[-1]
	tree_depth = np.iinfo(X.dtype).bits
	fbits = 1<<token_dim
	token = np.arange(1<<token_dim, dtype=np.uint8).reshape(-1,1)
	token = np.unpackbits(token, axis=-1)[:,-token_dim:]
	flags = BitBuffer(output + '.flg.bin')
	msg = "Layer: {:>2}, {:0>" + str(fbits) + "}, StackSize: {:>10}, Done: {:>3.2f}%"
	stack_size = 0
	
	if payload is True:
		payload = BitBuffer(output + '.pyl.bin') if output else BitBuffer()

	def expand(X, bits):
		flag = 0
		
		if len(X) == 0 or bits == 0:
			pass
		elif payload is not False and len(X) == 1:
			for d in range(token_dim):
				payload.write(int(X[:,d]), bits)
			payload.flush()
		else:
			for i, t in enumerate(token):
				m = np.all(X & 1 == t, axis=-1)
				if np.any(m):
					if bits > 1:
						yield expand(X[m] >> 1, bits-1)
					flag |= 1<<i
		if log.verbose:
			log(msg.format(tree_depth-bits, bin(flag)[2:], stack_size), end='\r', flush=True)
		flags.write(flag, fbits, soft_flush=True)
		pass
	
	nodes = deque(expand(X, tree_depth))
	while nodes:
		node = nodes.popleft() if breadth_first else nodes.pop()
		nodes.extend(node)
		stack_size = len(nodes)
	
	flags.close()
	if payload:
		payload.close()
	return flags, payload


def decode(flags, payload=None, xtype=np.uint16, breadth_first=False, **kwargs):
	if isinstance(payload, str):
		payload = BitBuffer(payload, 'rb')
	elif isinstance(payload, BitBuffer):
		payload.open(payload.name, 'rb')
	else:
		payload = None

	xtype = np.iinfo(xtype)
	fbits = np.iinfo(flags.dtype).bits
	dim = int(np.log2(fbits))
	token = np.arange(fbits, dtype=np.uint8).reshape(-1,1)
	token = np.unpackbits(token, axis=-1)[:,-dim:].astype(xtype)
	msg = "Layer: {:0>2}, {:0>" + str(fbits) + "}, Points: {:>10}, Done: {:>3.2f}%"
	done = np.zeros(1)
	points = np.zeros(1)
	ptr = iter(flags)
	X = np.zeros((np.sum(flags==0), dim), dtype=xtype)
	Xi = range(len(X))
	
	def expand(layer, x):
		flag = next(ptr, 0) if layer < xtype.bits else 0
		
		if flag == 0:
			if payload:
				x |= payload.read(xtype.bits - layer) << layer
			
			xi = next(Xi, None)
			if xi is not None:
				X[xi] = x
			else:
				X = np.vstack((X, x))
			points[:] += 1
			
		else:
			for bit in range(fbits):
				if flag & 1<<bit == 0:
					continue
				yield expand(layer+1, x | token[bit]<<layer)
		
		if log.verbose:
			done[:] += 1
			progress = 100.0 * float(done) / len(flags)
			log(msg.format(layer, bin(flag)[2:], int(points), progress), end='\r', flush=True)
		pass
		
	nodes = deque(expand(0, np.zeros(dim, dtype=xtype)))
	while nodes:
		nodes.extend(nodes.popleft() if breadth_first else nodes.pop())	
	return X


def load_datapoints(input_file, xtype=np.uint16, dim=3, **kwargs):
	X = np.fromfile(input_file, dtype=xtype)
	X = X[:(len(X)//args.dim)*args.dim].reshape(-1,args.dim)
	X = np.unique(X, axis=0)
	return X
	

def compress(X, **kwargs):
	log("\nData:", X.shape)
	log(X)
	log("\n---Encoding---\n")
	
	flags, payload = encode(X, **kwargs)
	
	log("Flags safed to:", flags.name)
	log("Payload safed to:", payload.name)
	return flags, payload


def load_flags(input_file, dim=3, **kwargs):
	if dim == 3:
		Y = np.fromfile(input_file, dtype=np.uint8)
	elif dim == 4:
		Y = np.fromfile(input_file, dtype=np.uint16)
	elif dim == 5:
		Y = np.fromfile(input_file, dtype=np.uint32)
	elif dim == 6:
		Y = np.fromfile(input_file, dtype=np.uint64)
	else:
		Y = BitBuffer(input_file, 'rb', 1<<dim)
	return Y


def decompress(Y, **kwargs):
	if output is None:
		output = flags
	if output:
		output = output.replace('.flg', '')
	
	log("\nFlags:", Y.shape)
	log(Y)
	log("\n---Decoding---\n")
	X = decode(flags, **kwargs)
	
	log("\nData:", X.shape)
	log(X)
	return X


def init_argparser(parents=[]):
	"""
	Initialize an ArgumentParser for this module.
	
	Args:
		parents: A list of ArgumentParsers of other scripts, if there are any.
		
	Returns:
		main_args: The ArgumentParsers.
	"""
	main_args = ArgumentParser(
		description="TokenTree",
		parents=parents
		)
	
	main_args.add_argument(
		'--output', '-o',
		metavar='PATH',
		default=None,
		help='A filename for the output data'
		)
	
	main_args.add_argument(
		'--xtype', '-t',
		metavar='TYPE',
		default='uint16',
		help='The expected data-type for the datapoints'
		)
	
	main_args.add_argument(
		'--dim', '-d',
		type=int,
		metavar='INT',
		default=3,
		help='The expected dimension of the datapoints'
		)
	
	main_args.add_argument(
		'--breadth_first', '-b',
		metavar='FLAG',
		nargs='?',
		type=bool,
		default=False,
		const=True,
		help='Flag whether the tree-structure is either breadth first or (default) depthfirst'
		)
	
	main_args.add_argument(
		'--verbose', '-v',
		metavar='FLAG',
		nargs='?',
		type=bool,
		default=False,
		const=True
		)
	
	subparsers = parser.add_subparsers(help='Application Mode')
	compress_args = parser.add_parser('compress', help='Compress datapoints to a TokenTree')
	decompress_args = parser.add_parser('decompress', help='Decompress a TokenTree to datapoints')
	
	compress_args.add_argument(
		'input_file',
		metavar='PATH',
		help='A path to a file of datapoints as .bin'
		)
	
	compress_args.add_argument(
		'--payload', '-p',
		metavar='FLAG',
		nargs='?',
		type=bool,
		default=False,
		const=True,
		help='Flag whether or (default) not to separate a payload file'
		)
	
	compress_args.set_defaults(
		load=load_datapoints,
		run=compress
		)
	
	decompress_args.add_argument(
		'input_file',
		metavar='PATH',
		help='A path to a file of branching flags as .flg.bin'
		)
	
	decompress_args.add_argument(
		'--payload', '-p',
		metavar='PATH',
		default=None,
		help='A path to a payload file of .ply.bin '
		)
	
	decompress_args.set_defaults(
		load=load_flags,
		run=decompress
		)
	
	return parser


def main(args):
	log.verbose = args.verbose
	
	if args.output is None:
		args.output = args.input_file
	if args.output:
		args.output = output.replace('.flg', '')
		args.output = output.replace('.bin', '')
	
	X = args.load(**args.__dict__)
	args.run(**args.__dict__)
	pass


if __name__ == '__main__':
	args, _ = init_argparser().parse_known_args()
	main(args)