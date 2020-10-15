#!/usr/bin/env python3

## Local
import mhdm.tokentree as tokentree


def compress(**kwargs):
	pass


def decompress(**kwargs):
	pass

decompress.add_argument(
		'--visualize', '-V',
		metavar='FLAG',
		nargs='?',
		type=bool,
		default=False,
		const=True,
		help='Flag whether the TokenTree starts from either heigher or (default) lower bit'
		)

compress_args.add_argument(
		'--sort_bits', '-B',
		metavar='FLAG',
		nargs='?',
		type=bool,
		default=False,
		const=True,
		help='Flag whether the bits of the datapoints get either sorted by probability or (default) not'
		)

main_args.add_argument(
		'--reverse', '-r',
		metavar='FLAG',
		nargs='?',
		type=bool,
		default=False,
		const=True,
		help='Flag whether the TokenTree starts from either heigher or (default) lower bit'
		)

decompress_args.add_argument(
		'--header', '-H',
		metavar='PATH',
		default=None,
		help='A path to a header file of hdr.bin'
		)
	
	decompress_args.add_argument(
		'--sort_bits', '-B',
		metavar='INT',
		nargs='*',
		type=int,
		default=[],
		help='A sequence of indices to rearange the bit order'
		)

def main(args):
	log.verbose = args.verbose
	if args.output:
		args.output = args.output.replace('.bin', '')
	header = None
	
	X = np.fromfile(args.compress, dtype=args.dtype)
	X = X[:(len(X)//args.dim)*args.dim].reshape(-1,args.dim)
	if args.sort_bits:
		X, p = sort_bits(X, args.reverse)
		log("\nBit order:", p)
		if args.output:
			header = args.output + '.hdr.bin'
			p.tofile(header)			
	elif args.reverse:
		X = reverse_bits(X)
	X = np.unique(X, axis=0)
	
	if log.verbose:
		log("\nData:", X.shape)
		log(X)
		log("\n---Encoding---\n")
	flags, payload = encode(X, **args.__dict__)
	
	log("Header safed to:", header)
	log("Flags safed to:", flags.name)
	log("Payload safed to:", payload.name)
	exit()
	np.concatenate((flags, payload), axis=None).tofile(args.output_file)
	
	log("\n---Decoding---\n")
	Y = Decoder(X.shape[-1]).expand(flags, payload.reshape(-1,X.shape[-1])).decode(X.dtype)
	log(Y)
	
	if args.visualize:
		import matplotlib.pyplot as plt
		import matplotlib.ticker as ticker
		from mpl_toolkits.mplot3d import Axes3D
	
		@ticker.FuncFormatter
		def major_formatter(i, pos):
			return "{:0>8}".format(bin(int(i))[2:])
		
		fig = plt.figure()
		ax = fig.add_subplot((111), projection='3d')
		ax.scatter(*X[:,:3].T, c=X.sum(axis=-1), s=0.5, alpha=0.5, marker='.')
		plt.show()
		
		ax = plt.subplot(111)
		ax.scatter(range(len(flags)), flags, 0.5, marker='.')
		ax.set_ylim(-7, 263)
		ax.yaxis.set_major_formatter(major_formatter)
		plt.show()


if __name__ == '__main__':
	args, _ = tokentree.init_argperser().parse_known_args()
	main(args)