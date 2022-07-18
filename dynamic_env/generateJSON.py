import json
import numpy as np

N_TYPES = ['empty', 'rigid', 'soft', 'hori', 'vert']
EMPTY_VX = 0
RIGID_VX = 1
SOFT_VX = 2
HORI_VX = 3
VERT_VX = 4
FIXED_VX = 5

def base_json(width, height):
    env_json = {
        "grid_width": width,
        "grid_height": height,
        "objects": {}
    }
    return env_json

def generate_env(width, height, barrier_interval, barrier_height_prob):
    env = np.zeros((height, width))
    env[0][:] = FIXED_VX # building the ground
    for j in range(12, width, barrier_interval):
        r = np.random.random()
        h = 1
        acc_prob = 0
        for prob in barrier_height_prob:
            acc_prob += prob
            if r <= acc_prob:
                break
            h += 1
        for i in range(0, h):
            env[i][j] = FIXED_VX
    return env

def ij_to_index(i, j, width):
    return j + i*width


def env2json(width=90, height=10, barrier_interval=1, barrier_height_prob=[.6,.2,.2]):
    ground = {
        "indices": [],
        "types": [],
        "neighbors": {}
    }
    env_vals = generate_env(width, height, barrier_interval, barrier_height_prob)
    for i in range(height):
        for j in range(width):
            idx = ij_to_index(i, j, width)
            vx_type = env_vals[i][j]
            if vx_type != EMPTY_VX:
                ground["indices"].append(idx)
                ground["types"].append(vx_type)
                neighbors = []
                if j > 0 and env_vals[i][j-1] != EMPTY_VX:
                    neighbors.append(ij_to_index(i, j-1, width))
                if j < width-1 and env_vals[i][j+1] != EMPTY_VX:
                    neighbors.append(ij_to_index(i, j+1, width))
                if i > 0 and env_vals[i-1][j] != EMPTY_VX:
                    neighbors.append(ij_to_index(i-1, j, width))
                if i < height-1 and env_vals[i+1][j] != EMPTY_VX:
                    neighbors.append(ij_to_index(i+1, j, width))
                ground["neighbors"][idx] = neighbors

    env_json = base_json(width, height)
    env_json["objects"]["ground"] = ground
    return env_json


def create_ObstacleTraverser_JSON():
    env = env2json()
    with open('dynamic_env/data.json', 'w', encoding='utf-8') as f:
        json.dump(env, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    create_ObstacleTraverser_JSON()