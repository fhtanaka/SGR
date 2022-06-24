from pickle import GLOBAL, POP
import neat
import os
import numpy as np
import errno
import dill
import neat
import time
import neat.nn
import pathlib

from typing import Dict, List, Tuple
from pathos.multiprocessing import ProcessPool
from evogym import hashable

from custom_reporter import CustomReporter, remove_reporters
from multiple_genome_arg_parser import parse_args
from generate_robot import generate_robot_CPPN_like
from evogym_sim import simulate_env, get_obs_size
from sgr_main import eval_genome_constraint, N_TYPES

BEST_FIT = -10000
STAG = 0
POPULATION = None

class  RobotController:
    def __init__(self, controllers) -> None:
        self.population = controllers
        self.best_fit = None
        self.evaluated_this_gen = False

def get_controller_config(robot, params):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, params["controller_config"])
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    
    in_size =  get_obs_size(robot, params)
    out_size = params["robot_size"]**2
    
    config.genome_config.num_inputs = in_size
    config.genome_config.input_keys = [-1*i for i in range(1, in_size+1)]

    config.genome_config.num_outputs = out_size
    config.genome_config.output_keys = [i for i in range(1, out_size+1)]

    config.pop_size = params["controller_pop"]
    return config

def get_controller_population(robot, params, robot_dict: Dict[str, RobotController]):
    robot_hash = hashable(robot)
    
    if robot_hash not in robot_dict:
        config = get_controller_config(robot, params)        
        p = neat.Population(config)
        robot_dict[robot_hash] = RobotController(p)

    return robot_dict[robot_hash]

def update_pop_fitness_thread(genomes, robot, control_neat_config, params):
    results_dict = {}
    for g_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, control_neat_config)
        fit, _ = simulate_env(robot, net, params)
        results_dict[g_id] = fit
    return results_dict

def controller_fit_func(genomes, control_neat_config, robot: np.array, params):

    try:
        pool = ProcessPool(nodes=params["cpu"])
        fit_func = lambda x: update_pop_fitness_thread(x, robot, control_neat_config, params)
        results = pool.map(
            fit_func,
            np.array_split(genomes, params["cpu"]),
        )

        fitness_dict = {}
        for result_dict in results:
            for k, v in result_dict.items():
                fitness_dict[k] = v

        for g_id, genome in genomes:
            genome.fitness = fitness_dict[g_id]

    except IOError as e:
        if e.errno == errno.EPIPE:
            print("Problem with broken pipe")
        else:
            raise(IOError)

def optimize_control(controller_pop, robot, params):
    c_fit_func = lambda genomes, config: controller_fit_func(genomes, config, robot, params)
    champion = controller_pop.run(c_fit_func, params["controller_in_between_gens"])
    return champion.fitness

def structure_fit_func(genomes, config, params, robot_dict: Dict[str, RobotController]):

    global BEST_FIT, STAG, POPULATION
    STAG += 1

    for robot in robot_dict.values():
        robot.evaluated_this_gen = False

    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        robot = generate_robot_CPPN_like(net, params["robot_size"])
        if not eval_genome_constraint(robot):
            genome.fitness = -10000
            continue

        robot_controllers = get_controller_population(robot, params, robot_dict)
        if robot_controllers.best_fit is not None and robot_controllers.evaluated_this_gen:
            genome.fitness = robot_controllers.best_fit
            continue
        
        best_fit = optimize_control(robot_controllers.population, robot, params)

        genome.fitness = best_fit
        robot_controllers.best_fit = best_fit
        robot_controllers.evaluated_this_gen = True

        if genome.fitness > BEST_FIT:
            BEST_FIT = genome.fitness
            STAG = 0
    
    if STAG > params["max_stag"]:
        print("!!!!!!!!!!!!!!!!!!!!! POPULATION STAGNATED !!!!!!!!!!!!!!!!!!!")
        if params["save_to"] is not "":
            dill.dump(POPULATION, open(params["save_to"] + "_pop.pkl", mode='wb'))
            dill.dump(robot_dict, open(params["save_to"] + "_robot_dict.pkl", mode='wb'))
        exit()


def main():
    params = parse_args()
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, params["structure_config"])
   

    neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    neat_config.pop_size = params["controller_pop"]

    pop = neat.Population(neat_config)

    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    if params["save_to"] is not "":
        pathlib.Path("/".join(params["save_to"].split("/")[:-1])).mkdir(parents=True, exist_ok=True) 
        pop.add_reporter(CustomReporter(True, params["save_to"] + "_out.txt", params["save_to"] + "_table.csv"))
    pop.add_reporter(neat.StdOutReporter(True))

    global POPULATION
    POPULATION = pop

    robot_dict = {}
    f = lambda genomes, config: structure_fit_func(genomes, config, params, robot_dict)
    winner = pop.run(f, params["gens"])
    print('\nBest genome:\n{!s}'.format(winner))

    if params["save_to"] is not "":
        remove_reporters(pop)
        dill.dump(pop, open(params["save_to"] + "_structure_pop.pkl", mode='wb'))
        dill.dump(robot_dict, open(params["save_to"] + "_robot_dict.pkl", mode='wb'))

if __name__ == "__main__":
    main()