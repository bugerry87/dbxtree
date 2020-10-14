
###TEST
if __name__ == '__main__':
    from argparse import ArgumentParser
    import numpy as np
    from trimesh import Trimesh
    from trimesh.proximity import closest_point
    from mhdm.utils import *
    
    def init_argparse(parents=[]):
        ''' init_argparse(parents=[]) -> parser
        Initialize an ArgumentParser for this module.
        
        Args:
            parents: A list of ArgumentParsers of other scripts, if there are any.
            
        Returns:
            parser: The ArgumentParsers.
        '''
        parser = ArgumentParser(
            #description="Test the trimesh distance",
            parents=parents
            )
        
        parser.add_argument(
            '--model_size', '-m',
            metavar='INT',
            type=int,
            default=30000
            )
        
        parser.add_argument(
            '--query_size', '-q',
            metavar='INT',
            type=int,
            default=30000
            )
        
        parser.add_argument(
            '--seed', '-s',
            metavar='INT',
            type=int,
            default=0
            )
        
        return parser
    
    args, _ = init_argparse().parse_known_args()
    
    delta = time_delta(time())
    np.random.seed(args.seed)
    X = np.random.randn(args.model_size,3)
    P = np.random.randn(args.query_size,3)
    Xi = np.arange(X.shape[0]).reshape(-1,3)
    
    print("trimesh.proximity")
    print("Model size:", X.shape)
    print("Query size:", P.shape)
    
    mesh = Trimesh(X, Xi, process=False)
    print("\nMesh setup time:", next(delta))
    
    closest, dist, tid = closest_point(mesh, P)
    
    print("\nQuery time:", next(delta))
    print("Mean loss:", dist.mean())
    
    print(closest)