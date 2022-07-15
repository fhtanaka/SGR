import itertools as it
import numpy as np
from evogym_sim import get_obs_size
from hyperneat.substrate import Substrate

def calc_layer(*coords):
    coord_arr = []
    for i in coords[:-1]:
        aux = np.linspace(-1.0, 1.0, i) if (i > 1) else [0.0]
        coord_arr.append(aux)
    last_coord = [coords[-1]]
    return tuple(it.product(*coord_arr, last_coord))

def morph_substrate_3D_out(params):
    design_in_layer = calc_layer(1, 1, 1, 1)
    design_h1 = calc_layer(params["robot_size"], params["robot_size"], 1, 2)
    design_out_layer = calc_layer(params["robot_size"], params["robot_size"], 5, 3)
    design_substrate = Substrate(design_in_layer, design_out_layer, [design_h1])
    return design_substrate


def morph_substrate_CPPN_like():
    design_in_layer = calc_layer(1, 2, 1)
    design_h1 = calc_layer(2, 2, 2)
    design_out_layer = calc_layer(1, 5, 3)
    design_substrate = Substrate(design_in_layer, design_out_layer, [design_h1])
    return design_substrate

def morph_substrate(params):
    return morph_substrate_3D_out(params)

def control_substrate_CPPN_like(params, robot):
    controller_in_layer = calc_layer(1, get_obs_size(robot, params), -1)
    controller_h1 = calc_layer(params["robot_size"], params["robot_size"], -2)
    controller_out_layer = calc_layer(params["robot_size"], params["robot_size"], -3)
    controller_substrate = Substrate(controller_in_layer, controller_out_layer, [controller_h1])
    return controller_substrate

def control_substrate_3D_out(params, robot):
    controller_in_layer = calc_layer(1, get_obs_size(robot, params), 1, -1)
    controller_h1 = calc_layer(params["robot_size"], params["robot_size"], 1, -2)
    controller_out_layer = calc_layer(params["robot_size"], params["robot_size"], 1, -3)
    controller_substrate = Substrate(controller_in_layer, controller_out_layer, [controller_h1])
    return controller_substrate


def control_substrate(params, robot):
    return control_substrate_3D_out(params, robot)
