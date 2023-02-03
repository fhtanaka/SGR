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

    graph = Graph(seed=10)
    nodes = []

    with open("grid_world/config.json", 'r', encoding='utf-8') as f:
        config_args = json.load(f)

    for _, n in config_args.items():
        _ = graph.add_node(n["task"], config_path, params)
        
    for n_id, n in config_args.items():
        for conn in n["neighbors"]:
            graph.connect_nodes_bidirectional(n_id, str(conn))
    
    for i in range(10):
        coord = str(graph.rng.integers(16))
        print("Evolving coord:", coord)
        graph.evolve_coord(coord, n_neighbors=4)

        path = f"cp_{i}.pkl"
        f = open(path, "wb")
        pickle.dump(graph, f)

if __name__ == "__main__":
    main()