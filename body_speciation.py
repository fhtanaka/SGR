import neat
import numpy as np
from hyperneat.new_hyperNEAT import create_phenotype_network
from neat.six_util import iteritems, iterkeys
from evogym import is_connected, has_actuator

def robot_from_genome(genome, params, substrate, robot_func, config):
    cppn = neat.nn.FeedForwardNetwork.create(genome, TempConfig(config))

    design_net = create_phenotype_network(cppn, substrate)
    robot = robot_func(design_net, params)
    
    if not eval_genome_constraint(robot):
        robot = np.zeros((params["robot_size"], params["robot_size"]))
    return robot

class TempConfig:
    def __init__(self, config):
        self.genome_config = config

def eval_genome_constraint(robot):
    validity = is_connected(robot) and has_actuator(robot)
    return validity

# This func was copy pasted from neat.DefaultGenome.distance
# Disnt reference directly to avoid recursion when calling form new_distance()
def old_distance(self: neat.DefaultGenome, other: neat.DefaultGenome, config):
    node_distance = 0.0
    if self.nodes or other.nodes:
        disjoint_nodes = 0
        for k2 in iterkeys(other.nodes):
            if k2 not in self.nodes:
                disjoint_nodes += 1

        for k1, n1 in iteritems(self.nodes):
            n2 = other.nodes.get(k1)
            if n2 is None:
                disjoint_nodes += 1
            else:
                # Homologous genes compute their own distance value.
                node_distance += n1.distance(n2, config)

        max_nodes = max(len(self.nodes), len(other.nodes))
        node_distance = (node_distance +
                            (config.compatibility_disjoint_coefficient *
                            disjoint_nodes)) / max_nodes

    # Compute connection gene differences.
    connection_distance = 0.0
    if self.connections or other.connections:
        disjoint_connections = 0
        for k2 in iterkeys(other.connections):
            if k2 not in self.connections:
                disjoint_connections += 1

        for k1, c1 in iteritems(self.connections):
            c2 = other.connections.get(k1)
            if c2 is None:
                disjoint_connections += 1
            else:
                # Homologous genes compute their own distance value.
                connection_distance += c1.distance(c2, config)

        max_conn = max(len(self.connections), len(other.connections))
        connection_distance = (connection_distance +
                                (config.compatibility_disjoint_coefficient *
                                disjoint_connections)) / max_conn

    distance = node_distance + connection_distance
    return distance

def new_distance(params, substrate, robot_func, self: neat.DefaultGenome, other: neat.DefaultGenome, config):

    if not hasattr(self, 'robot'):
        self.robot = robot_from_genome(self, params, substrate, robot_func, config)
    if not hasattr(other, 'robot'):
        other.robot = robot_from_genome(other, params, substrate, robot_func, config)

    diff = 0
    for i in range(params["robot_size"]):
        for j in range(params["robot_size"]):
            if (self.robot[i][j] == 0 and other.robot[i][j] != 0) or (self.robot[i][j] != 0 and other.robot[i][j] == 0):
                diff += 1
            elif self.robot[i][j] != other.robot[i][j]:
                diff += .5
            
    
    new_distance = diff/(params["robot_size"]**2) # Normalizing between 0 and 1
    original_dist = old_distance(self, other, config)

    return original_dist + 3*new_distance