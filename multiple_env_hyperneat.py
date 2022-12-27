
import os
import numpy as np
import sys
from typing import Dict
from pathos.multiprocessing import ProcessPool
from evogym import get_full_connectivity
import evogym.envs

from sgr.custom_reporter import CustomReporter, remove_reporters
from arg_parser import parse_args
from sgr.evogym_sim import get_obs_size
from sgr.generate_robot import eval_robot_constraint, N_TYPES
import os
from sgr.sgr import SGR
from dynamic_env.env_config import EnvConfig

N_ENVIRONMENTS = 6

def create_child(parent, rng, height_mutation_chance):
    seed = rng.integers(100)
    child = parent.create_child(seed)
    child.mutate_barrier_h(height_mutation_chance)
    return child


def main():
    params = parse_args()
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, params.neat_config)
    pop = SGR(
        config_path,
        params.robot_size,
        params.spec_genotype_weight,
        params.spec_phenotype_weight,
        params.pop_size,
        params.substrate_type,
        params.save_to,
        reporters=True
    )
    seed =  np.random.SeedSequence()
    rng = np.random.default_rng(seed)
    base_env = EnvConfig(seed)

    child_1 = create_child(base_env, rng, params.height_mutation_chance)
    child_2 = create_child(base_env, rng, params.height_mutation_chance)
    child_3 = create_child(child_1, rng, params.height_mutation_chance)
    child_4 = create_child(child_2, rng, params.height_mutation_chance)
    child_5 = create_child(child_3, rng, params.height_mutation_chance)


    env_bag = [base_env, child_1, child_2, child_3, child_4, child_5]

    # for _ in range(1, N_ENVIRONMENTS):
    #     parent_env = env_bag[rng.integers(0, len(env_bag))]
    #     seed = rng.integers(100)
    #     child = parent_env.create_child(seed)
    #     child.mutate_barrier_h(params.height_mutation_chance)
    #     print(parent_env.id, child.id, child.heights_list)
    #     env_bag.append(child)
    
    for _ in range(params.gens//params.p_transfer_gens):
        env_order: list[EnvConfig] = rng.choice(env_bag, 1, replace=False)
        for env in env_order:
            print("Training on env: ", env.id)
            pop.run(
                env_name="dynamic",
                n_steps=params.steps,
                n_gens=params.p_transfer_gens,
                cpus=params.cpu,
                max_stagnation=params.max_stag,
                save_gen_interval=params.save_gen_interval,
                print_results=False,
                dynamic_env_config=env,
            )

if __name__ == "__main__":
    main()