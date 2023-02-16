from .node import Node
from typing import Dict
from arg_parser import Parameters
from .tasks import *
import numpy as np
import copy
import neat
import pickle
from sgr.sgr import SGR
import json
import os
import pathlib
from sgr.body_speciation import CustomGenome
from generate_gif import generate_grid
from matplotlib import pyplot as plt
from tasks import get_locomotion_env_obs, get_manipulation_env_obs

RESULTS_DIR = "island_cp"

class HistoricalMarks:
    def __init__(self, genome_id, original_pop_id, parent1, parent2):
        self.id = genome_id
        self.pop_id = original_pop_id
        self.parent_1 = parent1
        self.parent_2 = parent2

class Graph:
    def __init__(self, seed, params: Parameters) -> None:
        self.d_nodes: Dict[str, Node] = {}
        self.d_historical: Dict[str, HistoricalMarks] = {}
        self.d_genomes: Dict[str, CustomGenome] = {}
        self.tasks = TaskList()
        self.rng = np.random.default_rng(seed)
        self.most_up_to_date_neat_pop: neat.Population = None
        self.params = params
        self.save_dir = ""
        self.report_file = None
        if self.params.save_to != "":
            self.save_dir = os.path.join(RESULTS_DIR, self.params.save_to)
            pathlib.Path(self.save_dir).mkdir(parents=True, exist_ok=True)
            self.report_file = open(os.path.join(self.save_dir, "report.txt"), "w+")
            # self.report_file.write("global_gen;pop_gen;pop_id;best_fit;num_species\n")

    def add_node(self, task, config_path, params: Parameters):
        pop = SGR(
            config_path,
            robot_size=params.robot_size,
            spec_genotype_weight=params.spec_genotype_weight,
            spec_phenotype_weight=params.spec_phenotype_weight,
            pop_size=params.pop_size,
            substrate_type=params.substrate_type,
            save_to="",
            reporters=False,
        )
        if self.most_up_to_date_neat_pop != None:
            neat_pop = copy.deepcopy(self.most_up_to_date_neat_pop)
            neat_pop.generation = 0
            neat_pop.best_genome = None
            neat_pop.population = neat_pop.reproduction.create_new(
                neat_pop.config.genome_type,
                neat_pop.config.genome_config,
                neat_pop.config.pop_size
            )
            neat_pop.species.speciate(neat_pop.config, neat_pop.population, neat_pop.generation)
            pop.pop = neat_pop

        for g in pop.pop.population.values():
            self.d_historical[g.key] = HistoricalMarks(g.key, pop.id, -1, -1)

        self.most_up_to_date_neat_pop = pop.pop
        task = self.tasks.task_dict[task]
        n = Node(pop, task.name, task.n_steps)
        self.d_nodes[str(n.id)] = n
        return str(n.id)

    def connect_nodes_bidirectional(self, n_id_1, n_id_2, dist=1):
        n1 = self.d_nodes[n_id_1]
        n2 = self.d_nodes[n_id_2]
        n1.add_conn(n_id_2, dist)
        n2.add_conn(n_id_1, dist)

    def log_to_file(self, node: Node, neighbors, winner):
        txt = f"Evolving coord: {node.id}, {node.task} for {node.n_steps} steps \n"
        txt += f"\tImporting pop from [{neighbors}] \n"
        txt += f"\tLocal gen {node.sgr_pop.pop.generation}, stag {node.sgr_pop.stagnation} \n"

        txt += f"\tBest fit ({winner.key}) from pop {self.d_historical[winner.key].pop_id}: {winner.fitness} \n"

        for g in node.sgr_pop.pop.population.values():
            if g.fitness != None:
                parents = node.sgr_pop.pop.reproduction.ancestors[g.key]
                txt += f"\t\t ag {g.key}, pop ({self.d_historical[g.key].pop_id}), parents ({parents}), fit {g.fitness}\n"

        txt += "\n\n"
        self.report_file.write(txt)

    def evolve_coord(self, node_id, max_dist=1, n_neighbors=4, repeat=False):

        main_node = self.d_nodes[node_id]
        print(f"Evolving coord: {main_node.id}, {main_node.task} for {main_node.n_steps} steps")

        current_neat_pop = main_node.sgr_pop.pop
        main_node.sgr_pop.pop = copy.deepcopy(self.most_up_to_date_neat_pop)
        
        main_pop = main_node.sgr_pop
        main_pop.pop.population = current_neat_pop.population
        main_pop.pop.best_genome = current_neat_pop.best_genome
        main_pop.pop.generation = current_neat_pop.generation

        possible_neighbors = [n_id for n_id, dist in main_node.connections.items() if dist <= max_dist]
        if len(possible_neighbors) <= n_neighbors and not repeat:
            neighbors = possible_neighbors
        else:            
            neighbors = self.rng.choice(possible_neighbors, n_neighbors, replace=repeat)

        for neighbor_id in neighbors:
            neighbor_genomes = self.d_nodes[neighbor_id].sgr_pop.pop.population
            for g in neighbor_genomes.values():
                main_pop.pop.population[g.key] = copy.deepcopy(g)

        main_pop.pop.species.speciate( main_pop.pop.config,  main_pop.pop.population,  main_pop.pop.generation)
        winner = main_pop.run(
            env_name = main_node.task, 
            n_steps = main_node.n_steps, 
            n_gens = 1, 
            cpus = self.params.cpu, 
            print_results = False, 
            get_env_obs = get_locomotion_env_obs # TODO: literal value
        )
        
        print(f"Local gen {main_pop.pop.generation}, stag {main_pop.stagnation}")
        print(f"Best fit ({winner.key}): {winner.fitness}")
        print()

        if self.report_file != None:
            self.log_to_file(main_node, neighbors, winner)

        self.most_up_to_date_neat_pop = main_pop.pop
        for g in main_pop.pop.population.values():
            if g.key not in self.d_historical:
                p1, p2 = main_pop.pop.reproduction.ancestors[g.key]
                self.d_historical[g.key] = HistoricalMarks(g.key, main_pop.id, p1, p2)
            if g.fitness != None and g.key not in self.d_genomes:
                self.d_genomes[g.key] = g


    def evolve_random_coords(self, n_gens):
        for i in range(n_gens):
            coord_id = str(self.rng.integers(len(self.d_nodes)))

            print("Gen ", i)
            if self.report_file != None:
                self.report_file.write(f"Gen {i}\n")
     
            self.evolve_coord(coord_id, n_neighbors=4)
            if self.params.save_to != "" and (
                i%self.params.save_gen_interval == 0 
                or (
                    self.d_nodes[coord_id].sgr_pop.stagnation == 0 and 
                    self.d_nodes[coord_id].sgr_pop.pop.generation > 1
                )
            ):
                self.save_grid_pkl(f"grid_gen_{i}.pkl")
                self.save_grid_img(f"gen_{i}_img.jpeg")


    def save_grid_pkl(self, file_name):
        temp_file, self.report_file = self.report_file, None 
        path = f"{self.save_dir}/{file_name}"
        f = open(path, "wb")
        pickle.dump(self, f)
        f.close()
        self.report_file = temp_file

    def save_grid_img(self, file_name):
        fig = generate_grid(self, 6, 6, self.params.substrate_type) # TODO: literal value, grid size
        fig.savefig(f"{self.save_dir}/{file_name}", pad_inches= 0.01)
        plt.clf()

    def interpret_json(self, file_name, neat_config_path):
        with open(file_name, 'r', encoding='utf-8') as f:
            config_args = json.load(f)

        for _, n in config_args.items():
            _ = self.add_node(n["task"], neat_config_path, self.params)
            
        for n_id, n in config_args.items():
            for conn in n["neighbors"]:
                self.connect_nodes_bidirectional(n_id, str(conn))