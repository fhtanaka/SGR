import os
import json
from arg_parser import parse_args
from sgr.sgr import SGR
from grid_world.graph import Graph
import pickle

def main():
    params = parse_args()
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, params.neat_config)

    graph = Graph(seed=params.seed, params=params)
    graph.interpret_json(params.task_grid, config_path)

    graph.evolve_random_coords(params.gens)

if __name__ == "__main__":
    main()