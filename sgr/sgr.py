from copy import deepcopy
from multiprocessing import TimeoutError
import multiprocess
import neat
import os
import numpy as np
import errno
import dill
import neat
import time
import neat.nn
import pathlib
import itertools
from neat.reporting import ReporterSet

from pathos.multiprocessing import ProcessPool

from hyperneat.new_hyperNEAT import create_phenotype_network
from sgr.custom_reporter import CustomReporter, remove_reporters
from sgr.body_speciation import CustomGenome
from sgr.substrates import morph_substrate, control_substrate
from sgr.generate_robot import generate_robot, eval_robot_constraint
from sgr.evogym_sim import simulate_env
from dynamic_env_v2.generateJSON import create_ObstacleTraverser_JSON

class SGR:
    idCounter = itertools.count().__next__

    def __init__(
            self,
            neat_config_path,
            robot_size,
            spec_genotype_weight,
            spec_phenotype_weight,
            pop_size,
            substrate_type,
            save_to="",
            reporters=True
        ):

        self.id = self.idCounter()
        morphology_coords = morph_substrate(robot_size, substrate_type)

        self.input_size = morphology_coords.dimensions*2 + 1 # two coordinates plus the bias
        self.pop_size = pop_size
        self.robot_size = robot_size
        self.substrate_type = substrate_type
        self.save_to = save_to

        CustomGenome.robot_func = lambda self, net, params: generate_robot(net, robot_size, substrate_type)
        CustomGenome.substrate = morphology_coords
        CustomGenome.robot_size = robot_size
        CustomGenome.spec_genotype_weight = spec_genotype_weight
        CustomGenome.spec_phenotype_weight = spec_phenotype_weight


        self.neat_config = self.create_neat_config(neat_config_path, CustomGenome)
        self.pop = neat.Population(self.neat_config)
        if reporters:
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
        neat_config.genome_config.num_outputs = 2
        neat_config.genome_config.output_keys = [1, 2]

        return neat_config

    def add_reporters(self):
        stats = neat.StatisticsReporter()
        self.pop.add_reporter(stats)
        if self.save_to is not "":
            pathlib.Path("/".join(self.save_to.split("/")[:-1])).mkdir(parents=True, exist_ok=True) 
            self.pop.add_reporter(CustomReporter(True, self.save_to + "_out.txt", self.save_to + "_table.csv"))
        self.pop.add_reporter(neat.StdOutReporter(True))

    def create_child(self):
        new_pop = deepcopy(self)
        new_pop.id = self.idCounter()
        new_pop.stagnation = 0
        new_pop.generation = 0
        new_pop.best_fit = -10000
        new_pop.best_genome = None
        new_pop.max_stagnation = None
        new_pop.save_gen_interval = None

        for _, ag in new_pop.pop.population.items():
            ag.fitness = None
        new_pop.pop.best_genome = None
        new_pop.pop.reporters = ReporterSet()
        new_pop.pop.generation = 0
        new_pop.pop.species = self.neat_config.species_set_type(self.neat_config.species_set_config, self.pop.reporters)
        new_pop.pop.species.speciate(self.neat_config, new_pop.pop.population, 0)

        return new_pop

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
            design_substrate = morph_substrate(self.robot_size, self.substrate_type)
            design_net = create_phenotype_network(cppn, design_substrate, output_node_idx=0)
            robot = generate_robot(design_net, self.robot_size, self.substrate_type)
            genome.robot = robot

        if not eval_robot_constraint(robot):
            return -10000, False

        try:
            controller_substrate = control_substrate(self.robot_size, env_name, robot, self.substrate_type)
        except IndexError: # Sometimes the environment just implodes
            return -10000, False

        controller_net = create_phenotype_network(cppn, controller_substrate, output_node_idx=1)

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
        local_dir = os.path.dirname(__file__)
        json_path = os.path.join(local_dir, "../dynamic_env/env.json")
        if env_name == "dynamic" and not os.path.exists(json_path):
            create_ObstacleTraverser_JSON(json_path)

        try:
            pool = ProcessPool(nodes=cpus)
            results_map = pool.amap(
                self.fit_func_thread,
                np.array_split(genomes, cpus),
                [n_steps for _ in range(cpus)],
                [env_name for _ in range(cpus)],
            )
            
            results = results_map.get(timeout=60*10)

            fitness_dict = {}
            for result_dict in results:
                for k, v in result_dict.items():
                    fitness_dict[k] = v

            for g_id, genome in genomes:
                genome.fitness = fitness_dict[g_id]
                if genome.fitness > self.best_fit:
                    self.best_fit = genome.fitness
                    self.stagnation = 0
                    self.best_genome = genome

        except IOError as e:  # Sometimes the environment just implodes
            if e.errno == errno.EPIPE:
                print("Problem with broken pipe")
            else:
                raise(IOError)
        except multiprocess.context.TimeoutError as e:
            print("Deu timeout!!!!!!")
            for g_id, genome in genomes:
                if genome.fitness is None:
                    genome.fitness = -1000

        pool.terminate()
        pool.clear()
        surviving_genomes = {g_id: genome for g_id, genome in genomes if genome.fitness is not None and genome.fitness > -1000}
        self.pop.population = surviving_genomes

        self.check_stagnation_and_save_interval()

    def check_stagnation_and_save_interval(self):
        # print("STAGNATION: ", self.stagnation)
        if self.max_stagnation is not None and self.stagnation > self.max_stagnation:
            print("!!!!!!!!!!!!!!!!!!!!! POPULATION STAGNATED !!!!!!!!!!!!!!!!!!!")
            if self.save_to is not "":
                dill.dump(self.pop, open(self.save_to + "_pop.pkl", mode='wb'))
            exit()
        if self.save_to is not "" and self.save_gen_interval is not None and (self.pop.generation+1)% self.save_gen_interval == 0:
            dill.dump(self.pop, open(f"{self.save_to}_pop_gen_{self.pop.generation}.pkl", mode='wb'))

    def run(self, env_name, n_steps, n_gens, cpus=1, max_stagnation=None, save_gen_interval=None, print_results=True):
        self.max_stagnation = max_stagnation
        self.save_gen_interval = save_gen_interval

        neat_fit_func = lambda genomes, config: self.fit_func(genomes, config, env_name, n_steps, cpus)
        winner: CustomGenome = self.pop.run(neat_fit_func, n_gens)
        
        if print_results:
            print('\nBest genome:\n{!s}'.format(winner))

        if self.save_to is not "":
            remove_reporters(self.pop)
            dill.dump(self.pop, open(self.save_to + "_pop.pkl", mode='wb'))
        
        return winner