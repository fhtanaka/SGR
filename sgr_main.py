import os
from arg_parser import parse_args
from sgr.sgr import SGR

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
        params.save_to
    )

    pop.run(
        params.env,
        params.steps,
        params.gens,
        params.cpu,
        params.max_stag,
        params.save_gen_interval
    )


if __name__ == "__main__":
    main()