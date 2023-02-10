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

    graph = Graph(seed=10, params=params)

    with open("grid_world/manipulation.json", 'r', encoding='utf-8') as f:
        config_args = json.load(f)

    for _, n in config_args.items():
        _ = graph.add_node(n["task"], config_path, params)
        
    for n_id, n in config_args.items():
        for conn in n["neighbors"]:
            graph.connect_nodes_bidirectional(n_id, str(conn))
    
    for i in range(5001):
        coord = str(graph.rng.integers(16))
        n = graph.d_nodes[coord]
        print("Gen ", i)
        print(f"Evolving coord: {coord}, {n.task} for {n.n_steps} steps")
        graph.evolve_coord(coord, n_neighbors=4)
        if i%5 == 0:
            path = f"island_cp/cp_manipulation_{i}.pkl"
            f = open(path, "wb")
            pickle.dump(graph, f)
        print(f"Local gen {n.sgr_pop.pop.generation}, stag {n.sgr_pop.stagnation}")
        print(f"Best fit ({n.sgr_pop.best_genome.key}): {n.sgr_pop.best_fit}")
        print()

if __name__ == "__main__":
    main()