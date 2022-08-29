import os
import numpy as np
import json
from generateJSON import env2json

def random_prob_distribution(size, rng: np.random.Generator):
    initial_values = [rng.random() for _ in range(size)]
    distribution = [n/sum(initial_values) for n in initial_values]
    return distribution

def random_prob_mutation(size, rng: np.random.Generator):
    initial_values = np.array([rng.random() for _ in range(size)])
    distribution = initial_values - initial_values.mean()
    return distribution

class EnvConfig:
    def __init__(self, seed, obstacle_height=0, obstacle_prob=np.array([1])):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.obs_h = obstacle_height
        self.heights_list = [n for n in range(-1*self.obs_h, self.obs_h+1)]
        self.obstacle_prob = obstacle_prob
    
    def randomize_config(self):
        self.obs_h = self.rng.integers(0,5)
        self.heights_list = [n for n in range(-1*self.obs_h, self.obs_h+1)]
        self.obstacle_prob = random_prob_distribution(len(self.heights_list), self.rng)

    def mutate_prob(self, mutation_power):
        mutation = random_prob_mutation(len(self.heights_list), self.rng)
        self.obstacle_prob += mutation * mutation_power
    
    def mutate_obs_h(self, max_mutation):
        possible_hs = []
        for i in range(self.obs_h - max_mutation, self.obs_h + max_mutation + 1):
            if -5 < i < 5 and i != self.obs_h:
                possible_hs.append(i)
        self.obs_h = self.rng.choice(possible_hs)
        self.heights_list = [n for n in range(-1*self.obs_h, self.obs_h+1)]
        self.obstacle_prob = random_prob_distribution(len(self.heights_list), self.rng)
        

    def generate_json(self, filename="data.json"):
        self.reset_seed()
        env = env2json(obstacle_height=self.heights_list, obstacle_prob=self.obstacle_prob, rng=self.rng)
        local_dir = os.path.dirname(__file__)
        path = os.path.join(local_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(env, f, ensure_ascii=False, indent=4)

    def generate_env_dict(self):
        self.reset_seed()
        return env2json(obstacle_height=self.heights_list, obstacle_prob=self.obstacle_prob, rng=self.rng)

    def reset_seed(self):
        self.rng = np.random.default_rng(self.seed)


if __name__ == "__main__":
    flat_env = EnvConfig(2)
    flat_env.generate_json("flat.json")

    env1 = EnvConfig(2)
    env1.randomize_config()
    env2 = EnvConfig(2, env1.obs_h, env1.obstacle_prob)
    
    env1.reset_seed()
    env1.generate_json("env1.json")
    env2.generate_json("env2.json")





    
