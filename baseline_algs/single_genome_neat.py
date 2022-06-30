import neat
import os
import numpy as np
import errno
import dill
import neat
import time
import neat.nn
import pathlib
import sys
from typing import Dict
from pathos.multiprocessing import ProcessPool
from evogym import get_full_connectivity
import evogym.envs

sys.path.append('../')
from custom_reporter import CustomReporter, remove_reporters
from alt_arg_parser import parse_args
from evogym_sim import get_obs_size
from sgr_main import eval_genome_constraint, N_TYPES

BEST_FIT = -10000
STAG = 0
POPULATION = None
OBS_SIZE = 0

def generate_robot(net, robot_size = 5, pad = 0):
    global OBS_SIZE
    robot = np.ones((robot_size, robot_size))
    for i in range(robot_size):
        for j in range(robot_size):
            input = (i - (robot_size // 2),
                     j - (robot_size // 2))
            pad = np.full(OBS_SIZE, pad)
            full_input = np.concatenate((input, pad))
            graph_out = net.activate(full_input)
            node = np.argmax(graph_out[:len(N_TYPES)])
            robot[i][j] = node
    return robot

def simulate_env(robot, net, config, render = False):
    connections = get_full_connectivity(robot)
    env = evogym.envs.gym.make(config["env"], body=robot, connections=connections)

    reward = 0

    obs = env.reset()
    actuators = env.get_actuator_indices("robot")
    
    global OBS_SIZE
    pre_pad = np.full(2, config["pad"])
    pos_pad = np.full(OBS_SIZE - len(obs), config["pad"])

    for _ in range(config["steps"]):
        if render:
            env.render('screen')
        input = np.concatenate((pre_pad, obs, pos_pad))

        action_by_actuator = net.activate(input)[len(N_TYPES):]
        action = np.array([action_by_actuator[i] for i in actuators]) 

        obs, r, done, _ = env.step(action)
        reward += r

        if done:
            return reward, True

    env.close()
    return reward, False

def fit_func_thread(genomes, params, neat_config, render=False, save_gif=False ):
    results_dict = {}
    for g_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, neat_config)
        robot = generate_robot(net, params["robot_size"], params["pad"])
        if not eval_genome_constraint(robot):
            results_dict[g_id] = -10000
            continue

        reward, _ = simulate_env(robot, net, params)
        results_dict[g_id] = reward
    return results_dict


def fit_func(genomes, neat_config, params):
    global BEST_FIT, STAG, POPULATION
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

        surviving_genomes = {}
        for g_id, genome in genomes:
            genome.fitness = fitness_dict[g_id]
            if genome.fitness > BEST_FIT:
                BEST_FIT = genome.fitness
                STAG = 0
            if genome.fitness > -10000:
                surviving_genomes[g_id] = genome

        POPULATION.population = surviving_genomes
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
            remove_reporters(POPULATION)
            dill.dump(POPULATION, open(params["save_to"] + "_pop.pkl", mode='wb'))
        exit()


def main():
    global POPULATION, OBS_SIZE

    params = parse_args()
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, params["neat_config"])
   
    neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    neat_config.pop_size = params["pop_size"]

    robot = np.full((params["robot_size"], params["robot_size"]), 4)
    OBS_SIZE = get_obs_size(robot, params)
    in_size =  2 + OBS_SIZE
    out_size = len(N_TYPES) + params["robot_size"]**2
    
    neat_config.genome_config.num_inputs = in_size
    neat_config.genome_config.input_keys = [-1*i for i in range(1, in_size+1)]
    # neat_config.genome_config.
    neat_config.genome_config.num_outputs = out_size
    neat_config.genome_config.output_keys = [i for i in range(1, out_size+1)]



    pop = neat.Population(neat_config)
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