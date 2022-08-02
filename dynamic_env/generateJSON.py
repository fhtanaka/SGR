import json
import numpy as np

N_TYPES = ['empty', 'rigid', 'soft', 'hori', 'vert']
EMPTY_VX = 0
RIGID_VX = 1
SOFT_VX = 2
HORI_VX = 3
VERT_VX = 4
FIXED_VX = 5

STARTING_ZONE = 12

def base_json(width, height):
    env_json = {
        "grid_width": width,
        "grid_height": height,
        "objects": {}
    }
    return env_json

def generate_env(width, height, barrier_height, barrier_prob):
    env = np.zeros((height, width))

    env[0:height//2][:STARTING_ZONE] = FIXED_VX # building the starting ground
    previous_height = height//2
    for j in range(STARTING_ZONE, width):
        r = np.random.choice(barrier_height, 1, p=barrier_prob)
        h = np.clip(previous_height + r, 1, height)
        for i in range(0, height):
            env[i][j] = EMPTY_VX
            if i < h:
                env[i][j] = FIXED_VX
        previous_height = h
    return env

def ij_to_index(i, j, width):
    return j + i*width

def add_neighbors(i, j, env_vals, width, height):
    neighbors = []
    if j > 0 and env_vals[i][j-1] != EMPTY_VX:
        neighbors.append(ij_to_index(i, j-1, width))
    if j < width-1 and env_vals[i][j+1] != EMPTY_VX:
        neighbors.append(ij_to_index(i, j+1, width))
    if i > 0 and env_vals[i-1][j] != EMPTY_VX:
        neighbors.append(ij_to_index(i-1, j, width))
    if i < height-1 and env_vals[i+1][j] != EMPTY_VX:
        neighbors.append(ij_to_index(i+1, j, width))
    return neighbors


def env2json(width=70, height=20, barrier_height=[-2,-1,0,1,2], barrier_prob=[.1, .25,.3,.25, .1]):
    ground = {
        "indices": [],
        "types": [],
        "neighbors": {}
    }
    env_vals = generate_env(width, height, barrier_height, barrier_prob)
    for i in range(height):
        for j in range(width):
            idx = ij_to_index(i, j, width)
            vx_type = env_vals[i][j]
            if vx_type != EMPTY_VX:
                ground["indices"].append(idx)
                ground["types"].append(vx_type)
                ground["neighbors"][idx] = add_neighbors(i, j, env_vals, width, height)

    env_json = base_json(width, height)
    env_json["objects"]["ground"] = ground
    return env_json


def create_ObstacleTraverser_JSON():
    env = env2json()
    with open('dynamic_env/data.json', 'w', encoding='utf-8') as f:
        json.dump(env, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    create_ObstacleTraverser_JSON()