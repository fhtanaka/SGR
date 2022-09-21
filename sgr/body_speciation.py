import neat
import numpy as np
from hyperneat.hyperNEAT import create_phenotype_network
from evogym import is_connected, has_actuator

def robot_from_genome(genome, robot_size, substrate, robot_func, config):
    cppn = neat.nn.FeedForwardNetwork.create(genome, TempConfig(config))

    design_net = create_phenotype_network(cppn, substrate)
    robot = robot_func(design_net, robot_size)
    
    if not (is_connected(robot) and has_actuator(robot)):
        robot = np.zeros((robot_size, robot_size))
    return robot

class TempConfig:
    def __init__(self, config):
        self.genome_config = config


class CustomGenome(neat.DefaultGenome):
    robot_size = None
    substrate = None
    robot_func = None
    spec_genotype_weight = None
    spec_phenotype_weight = None

    def __init__(self, key):
        super().__init__(key)
        if self.robot_size is None or self.substrate is None or self.robot_func is None or self.spec_genotype_weight is None or self.spec_phenotype_weight is None:
            print("Please define superparameters of CustomGenome")
            raise

    def distance(self, other, config):
        genotype_dist =  super().distance(other, config)

        if not hasattr(self, 'robot'):
            self.robot = robot_from_genome(self, self.robot_size, self.substrate, self.robot_func, config)
        if not hasattr(other, 'robot'):
            other.robot = robot_from_genome(other, self.robot_size, self.substrate, self.robot_func, config)

        diff = 0
        for i in range(self.robot_size):
            for j in range(self.robot_size):
                if (self.robot[i][j] == 0 and other.robot[i][j] != 0) or (self.robot[i][j] != 0 and other.robot[i][j] == 0):
                    diff += 1
                elif self.robot[i][j] != other.robot[i][j]:
                    diff += .5
                
        
        phenotype_dist = diff/(self.robot_size**2) # Normalizing between 0 and 1

        return self.spec_genotype_weight*genotype_dist + self.spec_phenotype_weight*phenotype_dist
