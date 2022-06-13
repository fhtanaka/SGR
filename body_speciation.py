import neat
import numpy as np
from hyperneat.new_hyperNEAT import create_phenotype_network
from neat.graphs import feed_forward_layers
from neat.six_util import itervalues
from evogym import is_connected, has_actuator

# def create_NN(genome, config):
#         """ Receives a genome and returns its phenotype (a FeedForwardNetwork). """

#         # Gather expressed connections.
#         connections = [cg.key for cg in itervalues(genome.connections) if cg.enabled]

#         layers = feed_forward_layers(config.input_keys, config.output_keys, connections)
#         node_evals = []
#         for layer in layers:
#             for node in layer:
#                 inputs = []
#                 node_expr = [] # currently unused
#                 for conn_key in connections:
#                     inode, onode = conn_key
#                     if onode == node:
#                         cg = genome.connections[conn_key]
#                         inputs.append((inode, cg.weight))
#                         node_expr.append("v[{}] * {:.7e}".format(inode, cg.weight))


#                 ng = genome.nodes[node]
#                 aggregation_function = config.aggregation_function_defs.get(ng.aggregation)
#                 activation_function = config.activation_defs.get(ng.activation)
#                 node_evals.append((node, activation_function, aggregation_function, ng.bias, ng.response, inputs))

#         return neat.nn.FeedForwardNetwork(config.input_keys, config.output_keys, node_evals)

def robot_from_genome(genome, params, substrate, robot_func, config):
    cppn = neat.nn.FeedForwardNetwork.create(genome, TempConfig(config))

    design_net = create_phenotype_network(cppn, substrate)
    robot = robot_func(design_net, params["robot_size"])
    
    if not eval_genome_constraint(robot):
        robot = np.zeros((params["robot_size"], params["robot_size"]))
    return robot

class TempConfig:
    def __init__(self, config):
        self.genome_config = config

def eval_genome_constraint(robot):
    validity = is_connected(robot) and has_actuator(robot)
    return validity

def new_distance(params, substrate, robot_func, self: neat.DefaultGenome, other: neat.DefaultGenome, config):

    if not hasattr(self, 'robot'):
        self.robot = robot_from_genome(self, params, substrate, robot_func, config)
    if not hasattr(other, 'robot'):
        other.robot = robot_from_genome(other, params, substrate, robot_func, config)

    diff = 0
    for i in range(params["robot_size"]):
        for j in range(params["robot_size"]):
            # if (self.robot[i][j] == 0 and other.robot[i][j] != 0) or (self.robot[i][j] != 0 and other.robot[i][j] == 0):
            if self.robot[i][j] != other.robot[i][j]:            
                diff += 1
            
    
    distance = diff/(params["robot_size"]**2)
    return distance