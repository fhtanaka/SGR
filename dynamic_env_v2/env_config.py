from copy import deepcopy
import os
import numpy as np
import json
import itertools
from .generateJSON import generate_env_json

class EnvConfig:
    idCounter = itertools.count().__next__

    def __init__(self, seed, width = 60, height = 10, flat_start = 8):
        self.id = self.idCounter()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.h = height
        self.w = width
        self.flat_start = flat_start
        self.heights_list = np.full((width), height//2)
   
    def mutate_barrier_h(self, mutation_prob):
        for idx, h in enumerate(self.heights_list):
            if idx < self.flat_start:
                pass 
            r = self.rng.random()
            if r < mutation_prob: 
                if r < mutation_prob/2:
                    h -= 1
                else:
                    h += 1
                self.heights_list[idx] = h      

    def generate_json(self, filename="env.json"):
        print(filename)
        env = generate_env_json(self.w, self.h, self.heights_list)
        local_dir = os.path.dirname(__file__)
        path = os.path.join(local_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(env, f, ensure_ascii=False, indent=4)

    def generate_env_dict(self):
        return generate_env_json(self.w, self.h, self.heights_list)

    def create_child(self, seed = None):
        child = deepcopy(self)
        child.id = self.idCounter()
        child.seed = self.rng.integers(100) if seed == None else seed
        self.rng = np.random.default_rng(child.seed)

        return child

if __name__ == "__main__":
    env1 = EnvConfig(1)
    env1.generate_json("env1.json")
    env = env1
    for i in range(10):
        new_env = env.create_child()
        new_env.mutate_barrier_h(.25)
        new_env.generate_json(f"env{i+2}.json")
        env = new_env




    
