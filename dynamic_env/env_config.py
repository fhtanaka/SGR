from copy import deepcopy
import os
import numpy as np
import json
import itertools
from .generateJSON import generate_env_json

def round_and_normalize_sum(arr):
    new_arr = np.array([np.clip(round(n, 2), 0, 1) for n in arr])
    if sum(new_arr) != 1:
        new_arr[np.argmax(new_arr)] += 1-sum(new_arr)
    return  new_arr

# Creates and array of length [size] with values [0, 1[ where the sum of elements is 1
def random_prob_distribution(size, rng: np.random.Generator):
    initial_values = [rng.random() for _ in range(size)]
    distribution = [n/sum(initial_values) for n in initial_values]
    return round_and_normalize_sum(distribution)

# Creates and array of length [size] with values [-1, 1[ where the sum of elements is 0
# This is used to mutate arrays without altering its sum
def random_prob_mutation(size, rng: np.random.Generator):
    initial_values = np.array([rng.random() for _ in range(size)])
    distribution = initial_values - initial_values.mean()
    return distribution

class EnvConfig:
    idCounter = itertools.count().__next__

    def __init__(self, seed, obstacle_height=0, obstacle_prob=np.array([1])):
        self.id = self.idCounter()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.barrier_h = obstacle_height
        self.heights_list = [n for n in range(-1*self.barrier_h, self.barrier_h+1)]
        self.obstacle_prob = obstacle_prob
    
    def randomize_config(self):
        self.barrier_h = self.rng.integers(0,5)
        self.heights_list = [n for n in range(-1*self.barrier_h, self.barrier_h+1)]
        self.obstacle_prob = random_prob_distribution(len(self.heights_list), self.rng)

    def mutate_obs_prob(self, mutation_power):
        mutation = random_prob_mutation(len(self.heights_list), self.rng)
        new_prob = self.obstacle_prob + (mutation * mutation_power)
        self.obstacle_prob = round_and_normalize_sum(new_prob)
    
    def mutate_barrier_h(self, max_mutation):
        possible_hs = []
        for i in range(self.barrier_h - max_mutation, self.barrier_h + max_mutation + 1):
            if 0 <= i < 5 and i != self.barrier_h:
                possible_hs.append(i)
        self.barrier_h = self.rng.choice(possible_hs)
        self.heights_list = [n for n in range(-1*self.barrier_h, self.barrier_h+1)]
        self.obstacle_prob = random_prob_distribution(len(self.heights_list), self.rng)
        

    def generate_json(self, filename="env.json"):
        temp_rng=np.random.default_rng(self.seed)
        self.obstacle_prob = round_and_normalize_sum(self.obstacle_prob)
        env = generate_env_json(obstacle_height=self.heights_list, obstacle_prob=self.obstacle_prob, rng=temp_rng)
        local_dir = os.path.dirname(__file__)
        path = os.path.join(local_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(env, f, ensure_ascii=False, indent=4)

    def generate_env_dict(self):
        temp_rng=np.random.default_rng(self.seed)
        return generate_env_json(obstacle_height=self.heights_list, obstacle_prob=self.obstacle_prob, rng=temp_rng)

    def create_child(self, seed = None):
        child = deepcopy(self)
        child.id = self.idCounter()
        child.seed = self.rng.integers(100) if seed == None else seed
        self.rng = np.random.default_rng(child.seed)

        return child

if __name__ == "__main__":
    env1 = EnvConfig(2, 1, [.3, .4, .3])
    for i in range(1000):
        env1.mutate_obs_prob(1)
        d = env1.generate_env_dict()




    
