from node import Node
from typing import Dict
from arg_parser import Parameters
from tasks import *

class Graph:
    def __init__(self) -> None:
        self.d_nodes: Dict[str, Node] = {}
        self.tasks = TaskList()

    def evolve_coord(self, node_id, dist=1):
        main_node = self.d_nodes[node_id]
        main_pop = main_node.sgr_pop.pop

        for neighbor_id in main_node.connections:
            node = self.d_nodes[neighbor_id]
            neighbor_pop = self.d_nodes[neighbor_id].sgr_pop.pop
            for g in neighbor_pop.population:
                # new_key = next(main_pop.reproduction.genome_indexer)
                main_pop.population[g.key] = g
         