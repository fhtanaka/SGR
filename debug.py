from poet.poet import POET
from arg_parser import parse_args
import pickle
import os
from sgr.body_speciation import CustomGenome
from sgr.substrates import morph_substrate
from sgr.generate_robot import generate_robot

import numpy as np

checkpoint = "/home/mma-lab/fabio/SGR/checkpoints/cp_5_3D_gen_1000.pkl"

def main():
    params = parse_args()
    file = open(checkpoint, mode='rb')
    poet: POET = pickle.load(file)
    poet.run_params.cpu = params.cpu
    
    CustomGenome.robot_func = lambda self, net, config: generate_robot(net, poet.run_params.robot_size)
    CustomGenome.substrate = morph_substrate(poet.run_params.robot_size)
    CustomGenome.robot_size = poet.run_params.robot_size
    CustomGenome.spec_genotype_weight = poet.run_params.spec_genotype_weight
    CustomGenome.spec_phenotype_weight = poet.run_params.spec_phenotype_weight

    poet.create_environments()

if __name__ == "__main__":
    main()