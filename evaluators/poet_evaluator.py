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
from sgr.body_speciation import CustomGenome
from poet.poet import POET

DIRS = [
    "new_cppn_1",
    "new_cppn_2",  
    "new_cppn_3",
    "new_3d_1",
    "new_3d_2",
    "new_3d_3",  
]

RESULTS_DIR = os.getcwd() + "/poet_results"
CPUS = 8
STEPS = 600


class POET_TEST:
    def __init__(self, test_name, test_env) -> None:
        self.test_name = test_name
        self.test_env = test_env
        self.csvs_dict = {}

        self.dir_path = f"{os.getcwd()}/../checkpoints/{test_name}"
        self.pkls = os.listdir(self.dir_path)
        self.pkls = [f for f in self.pkls if "csv" in f]
        self.pkls.sort()
        self.create_csv("global")

    def evaluate_gen(self, gen):
        file_path = f"{self.dir_path}/cp_{gen}.pkl"
        poet_pop: POET = pkl.load(open(file_path, "rb"))
        CustomGenome.robot_func = lambda self, net, config: generate_robot(net, poet_pop.run_params.robot_size)
        CustomGenome.substrate = morph_substrate(poet_pop.run_params.robot_size, poet_pop.run_params.substrate_type)
        CustomGenome.robot_size = poet_pop.run_params.robot_size
        CustomGenome.spec_genotype_weight = poet_pop.run_params.spec_genotype_weight
        CustomGenome.spec_phenotype_weight = poet_pop.run_params.spec_phenotype_weight

        global_results = []

        for p in poet_pop.pairs:
            pop = p.agent_pop
            env_id = p.environment.id
            if env_id not in self.csvs_dict:
                self.create_csv(env_id)
            pop.fit_func(
                genomes = [(g_id, g) for g_id, g in pop.pop.population.items()], 
                neat_config = pop.neat_config, 
                env_name = self.test_env, 
                n_steps = 600,
                cpus = CPUS
            ) 
            results = [a.fitness for _, a in pop.pop.population.items()]
            self.csvs_dict[env_id].write(f"{gen};{max(results)};{np.average(results)};{np.median(results)}\n")
            global_results = [*global_results, *results]
        
        self.csvs_dict["global"].write(f"{gen};{max(global_results)};{np.average(global_results)};{np.median(global_results)}\n")

    def create_csv(self, env_id):
        csv_file = f"{RESULTS_DIR}/{self.test_env}_{self.test_name}_{env_id}.csv"
        csv = open(csv_file, "w+")
        csv.write("gen;best_fit;avg_fit;median_fit\n")
        self.csvs_dict[env_id] = csv

if __name__ == "__main__":
    for t in ["Walker-v0", "ObstacleTraverser-v0"]:
        print("####################### ", t, "########################")
        for dir in DIRS:
            print("initiating test on: ", dir)
            p = POET_TEST(dir, t)
            for i in range(5, 202, 5):
                print(i, end=" ")
                p.evaluate_gen(i)
            print()