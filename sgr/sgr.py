import neat
import os
import numpy as np
import errno
import dill
import neat
import time
import neat.nn
import pathlib

from pathos.multiprocessing import ProcessPool

from hyperneat.new_hyperNEAT import create_phenotype_network
from sgr.custom_reporter import CustomReporter, remove_reporters
from sgr.body_speciation import CustomGenome
from sgr.substrates import morph_substrate, control_substrate
from sgr.generate_robot import generate_robot, eval_robot_constraint
from sgr.evogym_sim import simulate_env
from dynamic_env.generateJSON import create_ObstacleTraverser_JSON

class SGR:
    def __init__(
            self,
            neat_config_path,
            robot_size,
            spec_genotype_weight,
            spec_phenotype_weight,
            pop_size,
            save_to="",
        ):


        morphology_coords = morph_substrate(robot_size)

        self.input_size = morphology_coords.dimensions*2 + 1 # two coordinates plus the bias
        self.pop_size = pop_size
        self.robot_size = robot_size
        self.save_to = save_to

        CustomGenome.robot_func = lambda self, net, params: generate_robot(net, robot_size)
        CustomGenome.substrate = morphology_coords
        CustomGenome.robot_size = robot_size
        CustomGenome.spec_genotype_weight = spec_genotype_weight
        CustomGenome.spec_phenotype_weight = spec_phenotype_weight


        self.neat_config = self.create_neat_config(neat_config_path, CustomGenome)
        self.pop = neat.Population(self.neat_config)
        self.add_reporters()
        
        self.voxel_types = ['empty', 'rigid', 'soft', 'hori', 'vert']

        self.best_fit = -10000
        self.stagnation = 0
        self.generation = 0
        self.max_stagnation = None
        self.save_gen_interval = None

    def create_neat_config(self, config_path, neat_genome=neat.DefaultGenome):
        neat_config = neat.Config(neat_genome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
        
        # ovewriting pop_size from the neat config file
        neat_config.pop_size = self.pop_size

        # overwriting the num_inputs and num_outputs from the neat config file to fit the substrate
        neat_config.genome_config.num_inputs = self.input_size
        neat_config.genome_config.input_keys = [-1*i for i in range(1, self.input_size+1)]
        neat_config.genome_config.num_outputs = 1
        neat_config.genome_config.output_keys = [1]

        return neat_config

    def add_reporters(self):
        stats = neat.StatisticsReporter()
        self.pop.add_reporter(stats)
        if self.save_to is not "":
            pathlib.Path("/".join(self.save_to.split("/")[:-1])).mkdir(parents=True, exist_ok=True) 
            self.pop.add_reporter(CustomReporter(True, self.save_to + "_out.txt", self.save_to + "_table.csv"))
        self.pop.add_reporter(neat.StdOutReporter(True))


    def single_genome_fit(
            self, 
            genome,
            n_steps,
            env_name,
            render=False, 
            save_gif=None,
        ):

        cppn = neat.nn.FeedForwardNetwork.create(genome, self.neat_config)

        if hasattr(genome, 'robot'):
            robot = genome.robot
        else:
            design_substrate = morph_substrate(self.robot_size)
            design_net = create_phenotype_network(cppn, design_substrate)
            robot = generate_robot(design_net, self.robot_size)
            genome.robot = robot

        if not eval_robot_constraint(robot):
            return -10000, False

        try:
            controller_substrate = control_substrate(self.robot_size, env_name, robot)
        except IndexError: # Sometimes the environment just implodes
            return -10000, False

        controller_net = create_phenotype_network(cppn, controller_substrate)

        reward, done = simulate_env(robot, controller_net, env_name, n_steps, render, save_gif)

        return reward, done

    def fit_func_thread(self, genomes, n_steps, env_name):
        results_dict = {}
        for genome_key, genome in genomes:
            reward, _ = self.single_genome_fit(genome, n_steps, env_name)
            results_dict[genome_key] = reward
        return results_dict
    
    def fit_func(self, genomes, neat_config, env_name, n_steps, cpus):
        self.stagnation += 1
        start_t = time.time()
        # if env_name == "dynamic":
        #     create_ObstacleTraverser_JSON("../dynamic_env/data.json")

        try:
            pool = ProcessPool(nodes=cpus)
            results_map = pool.amap(
                self.fit_func_thread,
                np.array_split(genomes, cpus),
                [n_steps for _ in range(cpus)],
                [env_name for _ in range(cpus)],
            )
            
            results = results_map.get(timeout=60*60)

            fitness_dict = {}
            for result_dict in results:
                for k, v in result_dict.items():
                    fitness_dict[k] = v

            surviving_genomes = {}
            for g_id, genome in genomes:
                genome.fitness = fitness_dict[g_id]
                if genome.fitness > self.best_fit:
                    self.best_fit = genome.fitness
                    self.stagnation = 0
                if genome.fitness > -10000:
                    surviving_genomes[g_id] = genome

            self.pop.population = surviving_genomes

        except IOError as e:  # Sometimes the environment just implodes
            if e.errno == errno.EPIPE:
                print("Problem with broken pipe")
            else:
                raise(IOError)

        print("Simulation took ", time.time()-start_t, "s")
        self.check_stagnation_and_save_interval()

    def check_stagnation_and_save_interval(self):
        print("STAGNATION: ", self.stagnation)
        if self.max_stagnation is not None and self.stagnation > self.max_stagnation:
            print("!!!!!!!!!!!!!!!!!!!!! POPULATION STAGNATED !!!!!!!!!!!!!!!!!!!")
            if self.save_to is not "":
                dill.dump(self.pop, open(self.save_to + "_pop.pkl", mode='wb'))
            exit()
        if self.save_to is not "" and self.save_gen_interval is not None and (self.pop.generation+1)% self.save_gen_interval == 0:
            dill.dump(self.pop, open(f"{self.save_to}_pop_gen_{self.pop.generation}.pkl", mode='wb'))

    def run(self, env_name, n_steps, n_gens, cpus=1, max_stagnation=None, save_gen_interval=None):
        self.best_fit = -10000
        self.stagnation = 0
        self.generation = 0
        self.max_stagnation = max_stagnation
        self.save_gen_interval = save_gen_interval

        neat_fit_func = lambda genomes, config: self.fit_func(genomes, config, env_name, n_steps, cpus)
        winner = self.pop.run(neat_fit_func, n_gens)
        print('\nBest genome:\n{!s}'.format(winner))

        if self.save_to is not "":
            remove_reporters(self.pop)
            dill.dump(self.pop, open(self.save_to + "_pop.pkl", mode='wb'))
        
        return winner