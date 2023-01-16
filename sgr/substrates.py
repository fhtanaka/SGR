import itertools as it
import math
import numpy as np
from sgr.evogym_sim import get_obs_size
from hyperneat.substrate import Substrate

def raise_substrate_error():
    print("Substrate type should be specified")
    print("Available substrates: [cppn, 3d]")
    raise

def morph_substrate(robot_size, substrate_name):
    if substrate_name == "cppn" or substrate_name == "CPPN":
        shape = morph_substrate_CPPN_like_shape(robot_size)
    elif substrate_name == "3D" or substrate_name == "3d":
        shape = morph_substrate_3D_out_shape(robot_size)
    else:
        raise_substrate_error()
        
    return Substrate(shape)

def control_substrate(robot_size, env_name, robot, substrate_name):
    in_size = math.ceil(math.sqrt(get_obs_size(robot, env_name)))

    if substrate_name == "cppn" or substrate_name == "CPPN":
        shape = control_substrate_CPPN_like_shape(robot_size, in_size)
    elif substrate_name == "3D" or substrate_name == "3d":
        shape = control_substrate_3D_out_shape(robot_size, in_size)
    else:
        raise_substrate_error()

    return Substrate(shape)

def morph_substrate_3D_out_shape(robot_size):
    intermediate_layer = (1+robot_size)//2

    shape = [
        [1,1,1,1],
        [intermediate_layer, intermediate_layer, 3, 2],
        [robot_size, robot_size, 5, 3],
    ]
    return shape

def control_substrate_3D_out_shape(robot_size, in_size):
    intermediate_layer = (in_size+robot_size)//2
    shape = [
        [in_size, in_size, 1, -1],
        [intermediate_layer, intermediate_layer, 1, -2],
        [robot_size, robot_size, 1, -3]    
    ]
    return shape


def morph_substrate_CPPN_like_shape(robot_size):
    shape = [
        [1, 2, 1],
        [1, 3, 2],
        [1, 4, 3],
        [1, 5, 4]
    ]
    return shape

def control_substrate_CPPN_like_shape(robot_size, in_size):
    intermediate_layer = (in_size+robot_size)//2
    # [intermediate_layer, intermediate_layer, -2],
    shape = [
        [in_size, in_size, -1],
        [intermediate_layer, intermediate_layer, -2],
        [robot_size, robot_size, -3]
    ]
    return shape
