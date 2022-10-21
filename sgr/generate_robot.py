import numpy as np
from evogym import is_connected, has_actuator

N_TYPES = ['empty', 'rigid', 'soft', 'hori', 'vert']

def generate_robot_3D_out(net, robot_size):
    graph_out = net.activate([1997])
    formated_output = np.reshape(graph_out, (robot_size, robot_size, len(N_TYPES)), "F")
    robot = np.argmax(formated_output, 2)
    return robot


def generate_robot_CPPN_like(net, robot_size=5):
    robot = np.ones((robot_size, robot_size))
    for i in range(robot_size):
        for j in range(robot_size):
            input = [i - (robot_size // 2), j - (robot_size // 2)]
            # input = np.concatenate((input, [BIAS]))

            graph_out = net.activate(input)
            node = np.argmax(graph_out)
            robot[i][j] = node
    return robot

def generate_robot(net, robot_size):
    return generate_robot_3D_out(net, robot_size)
    # return premade_robot()

def premade_robot():
    a = [
        [3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3],
        [3, 3, 0, 3, 3],
        [3, 3, 0, 3, 3],
        [3, 3, 0, 3, 3],
    ]
    r = np.asarray(a)
    return r

def eval_robot_constraint(robot):
    validity = is_connected(robot) and has_actuator(robot)
    return validity