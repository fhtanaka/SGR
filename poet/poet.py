from distutils.command.config import config
from time import time
from typing import List
import numpy as np
from copy import deepcopy
import pickle
from dynamic_env.env_config import EnvConfig
from sgr.sgr import SGR
from arg_parser import Parameters
class Pair:
    """ A POET pair consisting of an environment and an agent. """
    def __init__(self, seed):
        self.environment = None
        self.agent_pop = None
        self.fitness = None
        self.seed = seed

    def init_first(self, params: Parameters, config_path):
        self.environment = EnvConfig(seed = self.seed)
        self.agent_pop = SGR(
            config_path,
            params.robot_size,
            params.spec_genotype_weight,
            params.spec_phenotype_weight,
            params.pop_size,
            params.save_to,
            reporters=True
        )

class POET:
    def __init__(self, seed: int, params: Parameters, config_path):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Parameters
        self.height_mutation_chance = params.height_mutation_chance 
        self.max_height_mutation = params.max_height_mutation 
        self.obs_prob_mutation_power = params.obs_prob_mutation_power 
        self.transfer_frequency = params.transfer_frequency 
        self.create_frequency = params.create_frequency 
        self.reproduction_criterion = params.reproduction_criterion 
        self.difficulty_criterion_low = params.difficulty_criterion_low
        self.difficulty_criterion_high = params.difficulty_criterion_high 
        self.num_create_environments = params.num_create_environments 
        self.num_children_add = params.num_children_add 
        self.max_pair_population_size = params.max_pair_population_size 
        self.n_nearest_neighbors = params.n_nearest_neighbors 

        
        # The pairs of environments and agents
        self.pairs: List[Pair] = []
        self.run_params = params
        self.config_path = config_path
        first_pair = Pair(self.rng.integers(100))
        first_pair.init_first(params, config_path)
        self.pairs.append(first_pair)

        # The archive with all environments that have ever existed in the pair population
        self.environment_archive = []
        self.environment_archive.append(first_pair.environment)

        self.total_environments_created = 1

    def run(self, generations):
        for i in range(1, generations):
            print("##################### Starting POET gen ", i, "#####################")
            print(f"Evaluating {len(self.pairs)} pairs\n")
            # Transfer
            if i%self.transfer_frequency == 0:
                print("Starting transfer process\n")
                self.transfer()
            # Create new environments
            if i%self.create_frequency == 0:
                print("Creating new environments\n")
                self.create_environments()
            # Train
            print("Population training\n")
            self.train_agents()
            # Create checkpoint
            if i%self.run_params.save_gen_interval == 0:
                self.save_checkpoint(i)
    def save_checkpoint(self, gen):
        path = "checkpoints/cp_cppn_gen_{}.pkl".format(gen)
        f = open(path, "wb")
        pickle.dump(self, f)
        f.close()

    def create_environments(self):
        # Find eligible pairs
        t = time()
        eligible_pairs = []
        for pair in self.pairs:
            if (pair.fitness is not None) and (pair.fitness > self.reproduction_criterion):
                eligible_pairs.append(pair)
        print("Eligible pairs to reproduce: ", len(eligible_pairs))
        # Create child environments
        child_environments = []
        if len(eligible_pairs) > 0:
            selected_pairs = np.random.choice(eligible_pairs, self.num_create_environments, replace=True)
            for pair in selected_pairs:
                child_environments.append(self.mutate(pair.environment))
        # Find agents for the children and test them against the minimal criteria
        eligible_child_pairs = []
        for environment in child_environments:
            child_pair = Pair(self.rng.integers(100))
            child_pair.environment = environment
            best_agent = None
            best_fitness = None
            for pair in self.pairs:
                child_pair.agent_pop = pair.agent_pop.create_child()
                child_pair.agent_pop.id = child_pair.agent_pop.idCounter()
                fitness = self.evaluate_pair(child_pair)
                if (best_fitness is None) or (fitness > best_fitness):
                    best_agent = child_pair.agent_pop
                    best_fitness = fitness
            if (best_fitness > self.difficulty_criterion_low) and (best_fitness < self.difficulty_criterion_high):
                child_pair.agent_pop = best_agent
                eligible_child_pairs.append(child_pair)
        # Select child environments to add to pair population
        sorted_child_pairs = self.sort_child_pairs(eligible_child_pairs)
        added = 0
        for child in sorted_child_pairs:
            if added < self.num_children_add:
                child.agent_pop.add_reporters()
                self.pairs.append(child)
                self.environment_archive.append(child.environment)
                if len(self.pairs) > self.max_pair_population_size:
                    self.pairs.pop(0)
            added += 1
        print(f"env creation took {time()-t}s\n")

    def mutate(self, env: EnvConfig):
        child = env.create_child()

        mutate_height = np.random.rand()
        if mutate_height and mutate_height < self.height_mutation_chance:
            child.mutate_barrier_h(self.max_height_mutation)
        else:
            child.mutate_obs_prob(self.obs_prob_mutation_power)    

        self.total_environments_created += 1
        return child
    
    # The difference from this and training is that this one only runs for 1 generations
    def evaluate_pair(self, pair: Pair, print_par_name = False):
        pop = pair.agent_pop
        env = pair.environment
        env.generate_json("env.json")
        if print_par_name:
            print(f"----- Env {pair.environment.id}, Pop {pair.agent_pop.id} -----")
        winner = pop.run(
            env_name = self.run_params.env,
            n_steps = self.run_params.steps,
            n_gens = 1,
            cpus = self.run_params.cpu,
            max_stagnation = self.run_params.max_stag,
            save_gen_interval = self.run_params.save_gen_interval,
            print_results = False
        )

        # Set fitness
        pair.fitness = winner.fitness
        return pair.fitness
    
    def sort_child_pairs(self, pairs: List[Pair]):
        # Remove already existing environments
        pruned_pairs = []
        for pair in pairs:
            if(not self.is_in_archive(pair.environment)):
                pruned_pairs.append(pair)
        # Compute novelty for the children
        novelties = []
        for pair in pruned_pairs:
            novelties.append(self.compute_novelty(pair.environment))
        # Sort children based on novelty
        sorted_pairs = []
        for i in range(len(novelties)):
            index = novelties.index(max(novelties))
            sorted_pairs.append(pruned_pairs.pop(index))
            novelties.pop(index)
        return sorted_pairs

    def is_in_archive(self, env):
        # Check if the environment already exists in the archive
        for environment in self.environment_archive:
            if self.compare_envs(environment, env) == 0:
                return True
        return False

    def compute_novelty(self, env):
        # Compute the novelty of an environment with regards to the archive
        # Novelty is the mean difference from the 5 nearest neighbours
        differences = []
        for environment in self.environment_archive:
            differences.append(self.compare_envs(environment, env))
        novelty = 0
        k = self.n_nearest_neighbors
        if len(differences) < k:
            k = len(differences)
        for i in range(k):
            novelty_i = min(differences)
            differences.pop(differences.index(novelty_i))
            novelty += novelty_i/k
        return novelty

    def compare_envs(self, env1: EnvConfig, env2: EnvConfig):
        # Find the difference between two environments
        diff_num = 0
        d = env1.barrier_h - env2.barrier_h
        diff_num += d if d>0 else -1*d
        if diff_num == 0:
            for i, j in zip(env1.obstacle_prob, env2.obstacle_prob):
                d = i-j
                diff_num += d if d>0 else -1*d
        return diff_num
    
    def train_agents(self):
        for pair in self.pairs:
            print(f"----------------- Env {pair.environment.id}, Pop {pair.agent_pop.id} -----------------")
            # Set environments
            pop = pair.agent_pop
            env = pair.environment
            env.generate_json("env.json")
            winner = pop.run(
                env_name = self.run_params.env,
                n_steps = self.run_params.steps,
                n_gens = self.run_params.gens,
                cpus = self.run_params.cpu,
                max_stagnation = self.run_params.max_stag,
                save_gen_interval = self.run_params.save_gen_interval,
                print_results=False
            )

            # Set fitness
            pair.fitness = winner.fitness
            print("Pair fitness: ", np.round(pair.fitness, 4), "\n")
            
    def transfer(self):
        # Direct transfer
        if len(self.pairs) >= 1:
            base_pairs = self.rng.choice(self.pairs, 3, replace=True)
            for pair in base_pairs:
                best_agent_pop = None
                best_fitness = -1000000
                for transfer_pair in self.pairs:
                    if transfer_pair.agent_pop.id != pair.agent_pop.id:
                        temp_test_pair = Pair(self.rng.integers(100))
                        temp_test_pair.environment = pair.environment
                        # if we remove the deepcopy in the line below, we will be doing the proposal transfer as well
                        # is this a good idea?
                        # temp_test_pair.agent_pop = deepcopy(transfer_pair.agent_pop)
                        temp_test_pair.agent_pop = transfer_pair.agent_pop
                        fitness = self.evaluate_pair(temp_test_pair, True)
                        if best_fitness < fitness:
                            best_agent_pop = temp_test_pair.agent_pop
                            best_fitness = fitness
                if best_fitness > pair.fitness:
                    pair.agent_pop = deepcopy(best_agent_pop)
                    pair.fitness = best_fitness
