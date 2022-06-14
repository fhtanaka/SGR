import multiprocessing
import neat
import os
import numpy as np
import errno
import dill
import neat
import time
import neat.nn
import pathlib

import pathos
from pathos.multiprocessing import ProcessPool
from evogym import is_connected, has_actuator

from hyperneat.new_hyperNEAT import create_phenotype_network
from custom_reporter import CustomReporter, remove_reporters
from arg_parser import parse_args
from body_speciation import new_distance
from substrates import morph_substrate, control_substrate
from generate_robot import generate_robot
from evogym_sim import simulate_env


N_TYPES = ['empty', 'rigid', 'soft', 'hori', 'vert']
BEST_FIT = -10000
STAG = 0
POPULATION = None

def eval_genome_constraint(robot):
    validity = is_connected(robot) and has_actuator(robot)
    return validity

def single_genome_fit(genome, params, neat_config, render=False):
    cppn = neat.nn.FeedForwardNetwork.create(genome, neat_config)

    if hasattr(genome, 'robot'):
        robot = genome.robot
    else:
        design_substrate = morph_substrate(params)
        design_net = create_phenotype_network(cppn, design_substrate)
        robot = generate_robot(design_net, params)

    if not eval_genome_constraint(robot):
        return -10000, False

    try:
        controller_substrate = control_substrate(params, robot)
    except IndexError: # Sometimes the environment just implodes
        return -10000, False

    controller_net = create_phenotype_network(cppn, controller_substrate)

    reward, done = simulate_env(robot, controller_net, params, render)

    genome.robot = robot
    return reward, done

def fit_func_thread(genomes, params, neat_config):
    results_dict = {}
    for genome_key, genome in genomes:
        reward, _ = single_genome_fit(genome, params, neat_config)
        results_dict[genome_key] = reward
    return results_dict

def fit_func(genomes, neat_config, params):
    global BEST_FIT, STAG
    STAG += 1
    start_t = time.time()
    try:
        pool = ProcessPool(nodes=params["cpu"])
        results_map = pool.amap(
            fit_func_thread,
            np.array_split(genomes, params["cpu"]),
            [params for _ in range(params["cpu"])],
            [neat_config for _ in range(params["cpu"])],
        )
        
        results = results_map.get(timeout=15*60)

        fitness_dict = {}
        for result_dict in results:
            for k, v in result_dict.items():
                fitness_dict[k] = v
        for g_id, genome in genomes:
            genome.fitness = fitness_dict[g_id]
            if genome.fitness > BEST_FIT:
                BEST_FIT = genome.fitness
                STAG = 0


    except IOError as e:  # Sometimes the environment just implodes
        if e.errno == errno.EPIPE:
            print("Problem with broken pipe")
        else:
            raise(IOError)
    print("Simulation took ", time.time()-start_t, "s")
    print("STAGNATION: ", STAG)
    if STAG > params["max_stag"]:
        print("!!!!!!!!!!!!!!!!!!!!! POPULATION STAGNATED !!!!!!!!!!!!!!!!!!!")
        if params["save_to"] is not "":
            dill.dump(genomes, open(params["save_to"] + "_genomes.pkl", mode='wb'))
            global POPULATION
            remove_reporters(POPULATION)
            dill.dump(POPULATION, open(params["save_to"] + "_population.pkl", mode='wb'))
        exit()

def main():
    params = parse_args()
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, params["neat_config"])
   
    defaultGen = neat.DefaultGenome
    f = lambda self, other, config: new_distance(params, morph_substrate(params), generate_robot, self, other, config)
    defaultGen.distance = f

    neat_config = neat.Config(defaultGen, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    neat_config.pop_size = params["pop_size"]

    pop = neat.Population(neat_config)
    global POPULATION
    POPULATION = pop
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    if params["save_to"] is not "":
        pathlib.Path("/".join(params["save_to"].split("/")[:-1])).mkdir(parents=True, exist_ok=True) 
        pop.add_reporter(CustomReporter(True, params["save_to"] + "_out.txt", params["save_to"] + "_table.csv"))
    pop.add_reporter(neat.StdOutReporter(True))

    f = lambda genomes, config: fit_func(genomes, config, params)
    winner = pop.run(f, params["gens"])
    print('\nBest genome:\n{!s}'.format(winner))

    if params["save_to"] is not "":
        remove_reporters(pop)
        dill.dump(pop, open(params["save_to"] + "_pop.pkl", mode='wb'))


if __name__ == "__main__":
    main()