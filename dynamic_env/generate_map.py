
import numpy as np
import json
from generateJSON import env2json


def generate_NxN_map(w_size):
    
    barriers = [-1,0,1]
    world = [[None for _ in range(w_size[1])] for _ in range(w_size[0]) ]
    for i in range(w_size[0]):
        for j in range(w_size[1]):
            minus_barrier = .1*(i+1)
            plus_barrier = .1*(j+1)
            no_barrier = 1-.1*(i+j+2)
            p = [minus_barrier, no_barrier, plus_barrier]
            w = env2json(barrier_height=barriers, barrier_prob=p)
            print(i, j)
            world[i][j] = w

    return world
if __name__ == "__main__":
    w_size = [5,5]
    w = generate_NxN_map(w_size)
    path = "dynamic_env/world"
    for i in range(w_size[0]):
        for j in range(w_size[1]):
            file = f"{path}/{i}x{j}.json"
            with open(file, 'w', encoding='utf-8') as f:
                json.dump(w[i][j], f, ensure_ascii=False, indent=4)