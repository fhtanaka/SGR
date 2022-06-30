import argparse

def parse_args():
    args_dict = {}

    # Default Values
    gens = 500
    robot_size = 5
    steps = 600
    env = "Walker-v0"     # env_names = ["CaveCrawler-v0", "UpStepper-v0", "ObstacleTraverser-v0"]
    n_threads = 6
    save_to = ""
    goal_fit = 10
    max_stag = 100
    
    structure_pop = 12
    controller_pop = 11
    controller_in_between_gens = 1
    pop_size = 64
    pad = 0
    neat_config = "../configs/hyperNEAT_config_CPPN_robot"





    neat_controller_config = "../configs/neat_controller.cfg"
    neat_structure_config = "../configs/neat_structure.cfg"

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gens", nargs="?", default=gens, help="", type=int) 
    parser.add_argument("-r", "--robot_size", nargs="?", default=robot_size, help="", type=int) 
    parser.add_argument("-s", "--steps", nargs="?", default=steps, help="", type=int) 
    parser.add_argument("-t", "--cpu", nargs="?", default=n_threads, help="", type=int) 
    parser.add_argument("-e", "--env", nargs="?", default=env, help="", type=str) 
    parser.add_argument("--save_to", nargs="?", default=save_to, help="", type=str) 
    parser.add_argument("--goal_fit", nargs="?", default=goal_fit, help="", type=float) 
    parser.add_argument("--max_stag", nargs="?", default=max_stag, help="", type=int)
    parser.add_argument("--pop", nargs="?", default=pop_size, help="", type=int) 
    parser.add_argument("--neat_config", nargs="?", default=neat_config, help="", type=str) 
    
    parser.add_argument("--controller_config", nargs="?", default=neat_controller_config, help="", type=str) 
    parser.add_argument("--structure_config", nargs="?", default=neat_structure_config, help="", type=str) 
    parser.add_argument("--controller_in_between_gens", nargs="?", default=controller_in_between_gens, help="", type=int)
    parser.add_argument("--structure_pop", nargs="?", default=structure_pop, help="", type=int)
    parser.add_argument("--controller_pop", nargs="?", default=controller_pop, help="", type=int)
    parser.add_argument("--pad", nargs="?", default=pad, help="", type=int)


    command_line_args = parser.parse_args()

    args_dict["gens"] = command_line_args.gens
    args_dict["pop_size"] = command_line_args.pop
    args_dict["robot_size"] = command_line_args.robot_size
    args_dict["steps"] = command_line_args.steps
    args_dict["env"] = command_line_args.env
    args_dict["cpu"] = command_line_args.cpu
    args_dict["save_to"] = command_line_args.save_to
    args_dict["goal_fit"] = command_line_args.goal_fit
    args_dict["max_stag"] = command_line_args.max_stag

    args_dict["controller_config"] = command_line_args.controller_config
    args_dict["structure_config"] = command_line_args.structure_config
    args_dict["controller_in_between_gens"] = command_line_args.controller_in_between_gens
    args_dict["structure_pop"] = command_line_args.structure_pop
    args_dict["controller_pop"] = command_line_args.controller_pop
    args_dict["pad"] = command_line_args.pad
    args_dict["neat_config"] = command_line_args.neat_config


    # if report is not None:
    for k, v in args_dict.items():
        print(f"{k}: {v}")
    print()

    return args_dict

