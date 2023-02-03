from .node import Node
from typing import Dict
from arg_parser import Parameters
from .tasks import *
import numpy as np
from sgr.sgr import SGR

class Graph:
    def __init__(self, seed) -> None:
        self.d_nodes: Dict[str, Node] = {}
        self.tasks = TaskList()
        self.rng = np.random.default_rng(seed)

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
        task = self.tasks.task_dict[task]
        n = Node(pop, task.name, task.n_steps)
        self.d_nodes[str(n.id)] = n
        return str(n.id)

    def connect_nodes_bidirectional(self, n_id_1, n_id_2, dist=1):
        n1 = self.d_nodes[n_id_1]
        n2 = self.d_nodes[n_id_2]
        n1.add_conn(n_id_2, dist)
        n2.add_conn(n_id_1, dist)

    def evolve_coord(self, node_id, max_dist=1, n_neighbors=4, repeat=False):
        main_node = self.d_nodes[node_id]
        main_pop = main_node.sgr_pop

        possible_neighbors = [n_id for n_id, dist in main_node.connections.items() if dist <= max_dist]
        neighbors = self.rng.choice(possible_neighbors, n_neighbors, replace=repeat)

        for neighbor_id in neighbors:
            neighbor_genomes = self.d_nodes[neighbor_id].sgr_pop.pop.population
            for g in neighbor_genomes.values():
                main_pop.pop.population[g.key] = g
        
        # Gambiarra: FIX later
        # This basically equals the indexer of the main_pop.pop.reproduction with the custom genome index
        # This is needed so the neat.Population.population dict stays in the format [genome.id, genome]
        current_id = [g for g in main_pop.pop.population.values()][0].idCounter()
        while next(main_pop.pop.reproduction.genome_indexer) < current_id:
            pass

        main_pop.pop.species.speciate( main_pop.pop.config,  main_pop.pop.population,  main_pop.pop.generation)
        main_pop.run(main_node.task, main_node.n_steps, 1, 4, print_results=False)
