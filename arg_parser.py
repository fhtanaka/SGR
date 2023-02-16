import argparse
import json
import os

def default_values():
    default = {
        "gens": 250,
        "robot_size": 5,
        "steps": 400,
        "env": "dynamic",     # env_names = ["CaveCrawler-v0", "UpStepper-v0", "ObstacleTraverser-v0"]
        "n_threads": 4,
        "save_to": "",
        "goal_fit": 10,
        "pop_size": 32,
        "max_stag": 10000,
        "neat_config": "neat_configs/hyperNEAT.cfg",
        "save_gen_interval": 20,
        "spec_genotype_weight": .8,
        "spec_phenotype_weight": 5,
        "substrate_type": "cppn",
        "seed": 1997,

        # used for POET, not required 
        "height_mutation_chance": 0.35,
        "max_height_mutation": 1,
        "obs_prob_mutation_power": 2,
        "reproduction_criterion": 1,
        "difficulty_criterion_low": .5,
        "difficulty_criterion_high": 8,
        "num_create_environments": 10,
        "num_children_add": 2,
        "max_pair_population_size": 20,
        "n_nearest_neighbors": 5,
        "p_transfer_gens": 1,

        "create_frequency": 49,
        "p_transfer_frequency": 10,
        "d_transfer_frequency": 27,

        # used for grid_world]
        "task_grid": "grid_world/grids/config.json",
    }

    return default

def create_parser(default_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", nargs="?", default="", help="", type=str)
    parser.add_argument("-g", "--gens", nargs="?", default=default_args["gens"], help="", type=int) 
    parser.add_argument("-r", "--robot_size", nargs="?", default=default_args["robot_size"], help="", type=int) 
    parser.add_argument("-s", "--steps", nargs="?", default=default_args["steps"], help="", type=int) 
    parser.add_argument("-t", "--cpu", nargs="?", default=default_args["n_threads"], help="", type=int) 
    parser.add_argument("-e", "--env", nargs="?", default=default_args["env"], help="", type=str) 
    parser.add_argument("--save_to", nargs="?", default=default_args["save_to"], help="", type=str) 
    parser.add_argument("--goal_fit", nargs="?", default=default_args["goal_fit"], help="", type=float) 
    parser.add_argument("--pop", nargs="?", default=default_args["pop_size"], help="", type=int) 
    parser.add_argument("--max_stag", nargs="?", default=default_args["max_stag"], help="", type=int)
    parser.add_argument("--neat_config", nargs="?", default=default_args["neat_config"], help="", type=str) 
    parser.add_argument("--save_gen_interval", nargs="?", default=default_args["save_gen_interval"], help="", type=int)
    parser.add_argument("--spec_genotype_weight", nargs="?", default=default_args["spec_genotype_weight"], help="", type=float) 
    parser.add_argument("--spec_phenotype_weight", nargs="?", default=default_args["spec_phenotype_weight"], help="", type=float) 
    parser.add_argument("--substrate", nargs="?", default=default_args["substrate_type"], help="", type=str) 
    parser.add_argument("--height_mutation_chance", nargs="?", default=default_args["height_mutation_chance"], help="", type=float)
    parser.add_argument("--max_height_mutation", nargs="?", default=default_args["max_height_mutation"], help="", type=int)
    parser.add_argument("--obs_prob_mutation_power", nargs="?", default=default_args["obs_prob_mutation_power"], help="", type=float)
    parser.add_argument("--create_frequency", nargs="?", default=default_args["create_frequency"], help="", type=int)
    parser.add_argument("--reproduction_criterion", nargs="?", default=default_args["reproduction_criterion"], help="", type=float)
    parser.add_argument("--difficulty_criterion_low", nargs="?", default=default_args["difficulty_criterion_low"], help="", type=float)
    parser.add_argument("--difficulty_criterion_high", nargs="?", default=default_args["difficulty_criterion_high"], help="", type=float)
    parser.add_argument("--num_create_environments", nargs="?", default=default_args["num_create_environments"], help="", type=int)
    parser.add_argument("--num_children_add", nargs="?", default=default_args["num_children_add"], help="", type=int)
    parser.add_argument("--max_pair_population_size", nargs="?", default=default_args["max_pair_population_size"], help="", type=int)
    parser.add_argument("--n_nearest_neighbors", nargs="?", default=default_args["n_nearest_neighbors"], help="", type=int)
    parser.add_argument("--p_transfer_gens", nargs="?", default=default_args["p_transfer_gens"], help="", type=int)
    parser.add_argument("--seed", nargs="?", default=default_args["seed"], help="", type=int)

    parser.add_argument("--p_transfer_frequency", nargs="?", default=default_args["p_transfer_frequency"], help="", type=int)
    parser.add_argument("--d_transfer_frequency", nargs="?", default=default_args["d_transfer_frequency"], help="", type=int)
    parser.add_argument("--task_grid", nargs="?", default=default_args["task_grid"], help="", type=str)

    return parser


def parse_args():
    args_dict = {}

    default_args = default_values()

    # Parsing just to change the default values in the case of a config file exists
    parser = create_parser(default_args)
    command_line_args = parser.parse_args()
    if command_line_args.config != "":
        local_dir = os.path.dirname(__file__)
        path = os.path.join(local_dir, command_line_args.config)
        with open(path, 'r', encoding='utf-8') as f:
            file_args = json.load(f)
            for k, v in file_args.items():
                default_args[k] = v

    # "real" parser to get the values from the command line that have priority over the
    # config file
    parser = create_parser(default_args)
    command_line_args = parser.parse_args()

    args_dict["gens"] = command_line_args.gens
    args_dict["robot_size"] = command_line_args.robot_size
    args_dict["steps"] = command_line_args.steps
    args_dict["env"] = command_line_args.env
    args_dict["cpu"] = command_line_args.cpu
    args_dict["save_to"] = command_line_args.save_to
    args_dict["goal_fit"] = command_line_args.goal_fit
    args_dict["pop_size"] = command_line_args.pop
    args_dict["max_stag"] = command_line_args.max_stag
    args_dict["neat_config"] = command_line_args.neat_config
    args_dict["save_gen_interval"] = command_line_args.save_gen_interval
    args_dict["spec_genotype_weight"] = command_line_args.spec_genotype_weight
    args_dict["spec_phenotype_weight"] = command_line_args.spec_phenotype_weight
    args_dict["substrate_type"] = command_line_args.substrate
    args_dict["seed"] = command_line_args.seed

    args_dict["height_mutation_chance"] = command_line_args.height_mutation_chance
    args_dict["max_height_mutation"] = command_line_args.max_height_mutation
    args_dict["obs_prob_mutation_power"] = command_line_args.obs_prob_mutation_power
    args_dict["create_frequency"] = command_line_args.create_frequency
    args_dict["reproduction_criterion"] = command_line_args.reproduction_criterion
    args_dict["difficulty_criterion_low"] = command_line_args.difficulty_criterion_low
    args_dict["difficulty_criterion_high"] = command_line_args.difficulty_criterion_high
    args_dict["num_create_environments"] = command_line_args.num_create_environments
    args_dict["num_children_add"] = command_line_args.num_children_add
    args_dict["max_pair_population_size"] = command_line_args.max_pair_population_size
    args_dict["n_nearest_neighbors"] = command_line_args.n_nearest_neighbors    
    args_dict["p_transfer_gens"] = command_line_args.p_transfer_gens
    args_dict["p_transfer_frequency"] = command_line_args.p_transfer_frequency
    args_dict["d_transfer_frequency"] = command_line_args.d_transfer_frequency

    args_dict["task_grid"] = command_line_args.task_grid

    return Parameters(args_dict)
    
class Parameters:
    def __init__(self, args_dict):
        self.gens = args_dict["gens"]
        self.robot_size = args_dict["robot_size"]
        self.steps = args_dict["steps"]
        self.env = args_dict["env"]
        self.cpu = args_dict["cpu"]
        self.save_to = args_dict["save_to"]
        self.goal_fit = args_dict["goal_fit"]
        self.pop_size = args_dict["pop_size"]
        self.max_stag = args_dict["max_stag"]
        self.neat_config = args_dict["neat_config"]
        self.save_gen_interval = args_dict["save_gen_interval"]
        self.spec_genotype_weight = args_dict["spec_genotype_weight"]
        self.spec_phenotype_weight = args_dict["spec_phenotype_weight"]
        self.substrate_type = args_dict["substrate_type"]
        self.seed = args_dict["seed"]

        self.p_transfer_gens = args_dict["p_transfer_gens"]
        self.height_mutation_chance = args_dict["height_mutation_chance"]
        self.max_height_mutation = args_dict["max_height_mutation"]
        self.obs_prob_mutation_power = args_dict["obs_prob_mutation_power"]
        self.create_frequency = args_dict["create_frequency"]
        self.reproduction_criterion = args_dict["reproduction_criterion"]
        self.difficulty_criterion_low = args_dict["difficulty_criterion_low"]
        self.difficulty_criterion_high = args_dict["difficulty_criterion_high"]
        self.num_create_environments = args_dict["num_create_environments"]
        self.num_children_add = args_dict["num_children_add"]
        self.max_pair_population_size = args_dict["max_pair_population_size"]
        self.n_nearest_neighbors = args_dict["n_nearest_neighbors"]

        self.p_transfer_frequency = args_dict["p_transfer_frequency"]
        self.d_transfer_frequency = args_dict["d_transfer_frequency"] 

        self.task_grid = args_dict["task_grid"] 
        # if report is not None:
        for k, v in args_dict.items():
            print(f"{k}: {v}")
        print()
