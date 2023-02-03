import os
from arg_parser import parse_args
from sgr.sgr import SGR
from grid_world.graph import Graph

def main():
    params = parse_args()
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, params.neat_config)

    graph = Graph(seed=10)
    nodes = []
    for _ in range(9):
        n_id = graph.add_node("Walker-v0", config_path, params)
        nodes.append(n_id)

    for i in range(9):
        if i == 4:
            continue
        graph.connect_nodes_bidirectional(i, 4)
    
    for i in range(5):
        graph.evolve_coord(4, n_neighbors=4)

if __name__ == "__main__":
    main()