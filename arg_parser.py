import argparse

def parse_args():
    args_dict = {}

    # Default Values
    gens = 250
    robot_size = 5
    steps = 400
    env = "dynamic"     # env_names = ["CaveCrawler-v0", "UpStepper-v0", "ObstacleTraverser-v0"]
    n_threads = 4
    save_to = ""
    goal_fit = 10
    pop_size = 32
    max_stag = 10000
    neat_config = "neat_configs/hyperNEAT.cfg"
    save_gen_interval = 20
    spec_genotype_weight = .8
    spec_phenotype_weight = 5

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gens", nargs="?", default=gens, help="", type=int) 
    parser.add_argument("-r", "--robot_size", nargs="?", default=robot_size, help="", type=int) 
    parser.add_argument("-s", "--steps", nargs="?", default=steps, help="", type=int) 
    parser.add_argument("-t", "--cpu", nargs="?", default=n_threads, help="", type=int) 
    parser.add_argument("-e", "--env", nargs="?", default=env, help="", type=str) 
    parser.add_argument("--save_to", nargs="?", default=save_to, help="", type=str) 
    parser.add_argument("--goal_fit", nargs="?", default=goal_fit, help="", type=float) 
    parser.add_argument("--pop", nargs="?", default=pop_size, help="", type=int) 
    parser.add_argument("--max_stag", nargs="?", default=max_stag, help="", type=int)
    parser.add_argument("--neat_config", nargs="?", default=neat_config, help="", type=str) 
    parser.add_argument("--save_gen_interval", nargs="?", default=save_gen_interval, help="", type=int)
    parser.add_argument("--spec_genotype_weight", nargs="?", default=spec_genotype_weight, help="", type=float) 
    parser.add_argument("--spec_phenotype_weight", nargs="?", default=spec_phenotype_weight, help="", type=float) 


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


    # if report is not None:
    for k, v in args_dict.items():
        print(f"{k}: {v}")
    print()

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

