from .node import Node
from typing import Dict
from arg_parser import Parameters
from .tasks import *
import numpy as np
import copy
import neat
from sgr.sgr import SGR

class Graph:
    def __init__(self, seed) -> None:
        self.d_nodes: Dict[str, Node] = {}
        self.tasks = TaskList()
        self.rng = np.random.default_rng(seed)
        self.most_up_to_date_neat_pop: neat.Population = None


    def add_node(self, task, config_path, params: Parameters):
        pop = SGR(
            config_path,
            params.robot_size,
            params.spec_genotype_weight,
            params.spec_phenotype_weight,
            params.pop_size,
            params.substrate_type,
            params.save_to,
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

    def evolve_coord(self, node_id, max_dist=1, n_neighbors=4, repeat=True):
        main_node = self.d_nodes[node_id]

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
        main_pop.run(main_node.task, main_node.n_steps, 1, 4, print_results=False)

        self.most_up_to_date_neat_pop = main_pop.pop
