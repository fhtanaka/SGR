import neat
import os
import evogym.envs
from evogym import is_connected, has_actuator, get_full_connectivity, hashable
import numpy as np
import pickle as pkl
import sys
sys.path.append('../')


from sgr.substrates import morph_substrate
from sgr.generate_robot import generate_robot
from sgr.sgr import SGR
from sgr.body_speciation import CustomGenome
from poet.poet import POET
from dynamic_env.env_config import EnvConfig

POET_DIRS = [
    "new_cppn_1",
    "new_cppn_2",  
    "new_cppn_3",
    "new_3d_1",
    "new_3d_2",
    "new_3d_3",  
]

MULT_ENV_FILES = [
    "multiple_env_cppn_1",
    "multiple_env_cppn_2",  
    "multiple_env_cppn_3",
    "multiple_env_3d_1",
    "multiple_env_3d_2",
    "multiple_env_3d_3",  
]

RESULTS_DIR = os.getcwd() + "/poet_results"
CPUS = 8
STEPS = 600


class POET_TEST:
    def __init__(self, test_name, envs) -> None:
        self.test_name = test_name
        self.csvs_dict = {}

        self.envs: list[EnvConfig] = envs
        self.dir_path = f"{os.getcwd()}/../checkpoints/{test_name}"
        self.create_csv("global")

    def evaluate_gen(self, gen):
        file_path = f"{self.dir_path}/cp_{gen}.pkl"
        poet_pop: POET = pkl.load(open(file_path, "rb"))
        CustomGenome.robot_func = lambda self, net, config: generate_robot(net, poet_pop.run_params.robot_size)
        CustomGenome.substrate = morph_substrate(poet_pop.run_params.robot_size, poet_pop.run_params.substrate_type)
        CustomGenome.robot_size = poet_pop.run_params.robot_size
        CustomGenome.spec_genotype_weight = poet_pop.run_params.spec_genotype_weight
        CustomGenome.spec_phenotype_weight = poet_pop.run_params.spec_phenotype_weight

        for p in poet_pop.pairs:
            pop = p.agent_pop
            original_env_id = p.environment.id
            winner = p.agent_pop.pop.best_genome
            for env in self.envs: 
                reward, _ = pop.single_genome_fit(
                    genome = winner,
                    n_steps = STEPS,
                    env_name = "dynamic",
                    dynamic_env_config = env
                )
                print("   ", original_env_id, env.id, reward)
                results = f"{gen}; {original_env_id}; {env.id};{reward}\n"
                self.csvs_dict["global"].write(results)

    def create_csv(self, original_env_id):
        csv_file = f"{RESULTS_DIR}/POET_{self.test_name}_{original_env_id}.csv"
        csv = open(csv_file, "w+")
        csv.write("gen;original_env_id;test_env_id;fitness\n")
        self.csvs_dict[original_env_id] = csv

class MULT_ENV_TEST:
    def __init__(self, test_name, envs) -> None:
        self.test_name = test_name
        self.csvs_dict = {}

        self.envs: list[EnvConfig] = envs
        self.dir_path = f"{os.getcwd()}/../multiple_env_results/{test_name}"
        self.create_csv("global")

    def evaluate_gen(self, gen):
        file_path = f"{self.dir_path}_pop_gen_{gen}.pkl"
        pop: SGR = pkl.load(open(file_path, "rb"))
        CustomGenome.robot_func = lambda self, net, config: generate_robot(net, pop.robot_size)
        CustomGenome.substrate = morph_substrate(pop.robot_size, pop.substrate_type)
        CustomGenome.robot_size = pop.robot_size
        CustomGenome.spec_genotype_weight = 1
        CustomGenome.spec_phenotype_weight = 2

        winner = pop.pop.best_genome
        for env in self.envs: 
            reward, _ = pop.single_genome_fit(
                genome = winner,
                n_steps = STEPS,
                env_name = "dynamic",
                dynamic_env_config = env
            )
            print("   ", 0, env.id, reward)
            results = f"{gen}; {0}; {env.id};{reward}\n"
            self.csvs_dict["global"].write(results)

    def create_csv(self, original_env_id):
        csv_file = f"{RESULTS_DIR}/MULT_ENV_{self.test_name}_{original_env_id}.csv"
        csv = open(csv_file, "w+")
        csv.write("gen;original_env_id;test_env_id;fitness\n")
        self.csvs_dict[original_env_id] = csv


if __name__ == "__main__":
    env0 = EnvConfig(0)
    env1 = env0.create_child()
    env1.mutate_barrier_h(.25)
    env2 = env0.create_child()
    env2.mutate_barrier_h(.25)
    env3 = env1.create_child()
    env3.mutate_barrier_h(.25)
    env4 = env2.create_child()
    env4.mutate_barrier_h(.25)

    envs = [env0, env1, env2, env3, env4]


    for dir in POET_DIRS:
        print("initiating test on: ", dir)
        p = POET_TEST(dir, envs)
        for i in range(5, 202, 5):
            print(i)
            p.evaluate_gen(i)
        print()
        for f in p.csvs_dict.values():
            f.close()

    for f in MULT_ENV_FILES:
        print("initiating test on: ", f)
        p = MULT_ENV_TEST(f, envs)
        for i in range(5, 202, 5):
            print(i)
            p.evaluate_gen(i)
        print()
        for f in p.csvs_dict.values():
            f.close()