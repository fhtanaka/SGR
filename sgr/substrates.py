import itertools as it
import math
import numpy as np
from sgr.evogym_sim import get_obs_size
from hyperneat.substrate import Substrate

def morph_substrate(robot_size):
    shape = morph_substrate_CPPN_like_shape(robot_size)
    return Substrate(shape)

def control_substrate(robot_size, env_name, robot):
    shape = control_substrate_CPPN_like_shape(robot_size, env_name, robot)
    return Substrate(shape)

def morph_substrate_3D_out_shape(robot_size):
    shape = [
        [1,1,1,1],
        [robot_size//2, robot_size//2, 3, 2],
        [robot_size, robot_size, 5, 3],
    ]
    return shape

def control_substrate_3D_out_shape(robot_size, env_name, robot):
    in_size = math.ceil(math.sqrt(get_obs_size(robot, env_name)))
    shape = [
        [in_size, in_size, 1, -1],
        [robot_size, robot_size, 1, -2],
        [robot_size, robot_size, 1, -3]    
    ]
    return shape


def morph_substrate_CPPN_like_shape(robot_size):
    shape = [
        [1, 2, 1],
        [1, 4, 2],
        [1, 5, 3]
    ]
    return shape

def control_substrate_CPPN_like_shape(robot_size, env_name, robot):
    in_size = math.ceil(math.sqrt(get_obs_size(robot, env_name)))
    shape = [
        [in_size, in_size, -1],
        [robot_size, robot_size, -2],
        [robot_size, robot_size, -3]
    ]
    return shape
