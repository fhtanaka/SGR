from poet.poet import POET
from arg_parser import parse_args
from sgr.sgr import SGR
import os
import numpy as np

def main():
    params = parse_args()

    params.gens = 8
    params.robot_size = 5
    params.steps = 450
    params.env = "dynamic"
    params.cpu = 8
    params.pop_size = 32
    params.neat_config="configs/POET.cfg"
    # params.save_to = "results_poet/test"
    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, params.neat_config)
    seed =  np.random.SeedSequence()
    
    aux = POET(
        seed,
        params,
        config_path
    )

    aux.run(100)


if __name__ == "__main__":
    main()