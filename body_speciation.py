import neat
import numpy as np
from hyperneat.hyperNEAT import create_phenotype_network
from evogym import is_connected, has_actuator

def robot_from_genome(genome, params, substrate, robot_func, config):
    cppn = neat.nn.FeedForwardNetwork.create(genome, TempConfig(config))

    design_net = create_phenotype_network(cppn, substrate)
    robot = robot_func(design_net, params)
    
    if not (is_connected(robot) and has_actuator(robot)):
        robot = np.zeros((params["robot_size"], params["robot_size"]))
    return robot

class TempConfig:
    def __init__(self, config):
        self.genome_config = config


class CustomGenome(neat.DefaultGenome):
    params = None
    substrate = None
    robot_func = None

    def __init__(self, key):
        super().__init__(key)
    def distance(self, other, config):
        genotype_dist =  super().distance(other, config)

        if not hasattr(self, 'robot'):
            self.robot = robot_from_genome(self, self.params, self.substrate, self.robot_func, config)
        if not hasattr(other, 'robot'):
            other.robot = robot_from_genome(other, self.params, self.substrate, self.robot_func, config)

        diff = 0
        for i in range(self.params["robot_size"]):
            for j in range(self.params["robot_size"]):
                if (self.robot[i][j] == 0 and other.robot[i][j] != 0) or (self.robot[i][j] != 0 and other.robot[i][j] == 0):
                    diff += 1
                elif self.robot[i][j] != other.robot[i][j]:
                    diff += .5
                
        
        phenotype_dist = diff/(self.params["robot_size"]**2) # Normalizing between 0 and 1

        return self.params["spec_genotype_weight"]*genotype_dist + self.params["spec_phenotype_weight"]*phenotype_dist
