import itertools as it
import numpy as np
from evogym_sim import get_obs_size
from hyperneat.substrate import Substrate

def morph_substrate(params):
    shape = morph_substrate_CPPN_like_shape(params)
    return Substrate(shape)

def control_substrate(params, robot):
    shape = control_substrate_CPPN_like_shape(params, robot)
    return Substrate(shape)

def morph_substrate_3D_out_shape(params):
    shape = [
        [1,1,1,1],
        [params["robot_size"]//2, params["robot_size"]//2, 3, 2],
        [params["robot_size"], params["robot_size"], 5, 3],
    ]
    return shape

def control_substrate_3D_out_shape(params, robot):
    shape = [
        [1, get_obs_size(robot, params), 1, -1],
        [params["robot_size"], params["robot_size"], 1, -2],
        [params["robot_size"], params["robot_size"], 1, -3]    
    ]
    return shape


def morph_substrate_CPPN_like_shape(params):
    shape = [
        [1, 2, 1],
        [1, 4, 2],
        [1, 5, 3]
    ]
    return shape

def control_substrate_CPPN_like_shape(params, robot):
    shape = [
        [1, get_obs_size(robot, params), -1],
        [params["robot_size"], params["robot_size"], -2],
        [params["robot_size"], params["robot_size"], -3]
    ]
    return shape
