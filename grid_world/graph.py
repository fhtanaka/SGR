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
        self.tasks = TaskList()
        self.rng = np.random.default_rng(seed)
        self.most_up_to_date_neat_pop: neat.Population = None
        self.params = params
        self.save_dir = ""
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
            reporters=False
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

    def evolve_coord(self, node_id, max_dist=1, n_neighbors=4, repeat=True, print_info = True):

        main_node = self.d_nodes[node_id]
        print(f"Evolving coord: {node_id}, {main_node.task} for {main_node.n_steps} steps")

        current_neat_pop = main_node.sgr_pop.pop
        main_node.sgr_pop.pop = copy.deepcopy(self.most_up_to_date_neat_pop)
        
        main_pop = main_node.sgr_pop
        main_pop.pop.population = current_neat_pop.population
        main_pop.pop.best_genome = current_neat_pop.best_genome
        main_pop.pop.generation = current_neat_pop.generation

        possible_neighbors = [n_id for n_id, dist in main_node.connections.items() if dist <= max_dist]
        neighbors = self.rng.choice(possible_neighbors, n_neighbors, replace=repeat)

        for neighbor_id in neighbors:
            neighbor_genomes = self.d_nodes[neighbor_id].sgr_pop.pop.population
            for g in neighbor_genomes.values():
                main_pop.pop.population[g.key] = g

        main_pop.pop.species.speciate( main_pop.pop.config,  main_pop.pop.population,  main_pop.pop.generation)
        winner = main_pop.run(main_node.task, main_node.n_steps, 1, self.params.cpu, print_results=False)
        
        print(f"Local gen {main_pop.pop.generation}, stag {main_pop.stagnation}")
        print(f"Best fit ({winner.key}): {winner.fitness}")
        print()

        self.most_up_to_date_neat_pop = main_pop.pop
        for g in main_pop.pop.population.values():
            if g.key in self.d_historical:
                continue
            p1, p2 = main_pop.pop.reproduction.ancestors[g.key]
            self.d_historical[g.key] = HistoricalMarks(g.key, main_pop.id, -1, -1)

    def evolve_random_coords(self, n_gens):
        for i in range(n_gens):
            coord_id = str(self.rng.integers(len(self.d_nodes)))
            n = self.d_nodes[coord_id]

            print("Gen ", i)
            self.evolve_coord(coord_id, n_neighbors=4)
            if self.params.save_to != "" and i%self.params.save_gen_interval == 0:
                temp_file, self.report_file = self.report_file, None 
                path = f"{self.save_dir}/grid_gen_{i}.pkl"
                f = open(path, "wb")
                pickle.dump(self, f)
                f.close()
                self.report_file = temp_file

    # def write_report

    def interpret_json(self, file_name, neat_config_path):
        with open(file_name, 'r', encoding='utf-8') as f:
            config_args = json.load(f)

        for _, n in config_args.items():
            _ = self.add_node(n["task"], neat_config_path, self.params)
            
        for n_id, n in config_args.items():
            for conn in n["neighbors"]:
                self.connect_nodes_bidirectional(n_id, str(conn))