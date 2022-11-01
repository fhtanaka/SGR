from poet.poet import POET
from arg_parser import parse_args
from sgr.sgr import SGR
import os
import numpy as np

def main():
    params = parse_args()
    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, params.neat_config)
    seed =  np.random.SeedSequence()
    
    poet_alg = POET(
        seed,
        params,
        config_path
    )

    poet_alg.run(params.gens)


if __name__ == "__main__":
    main()