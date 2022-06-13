import numpy as np

N_TYPES = ['empty', 'rigid', 'soft', 'hori', 'vert']

def generate_robot_in_1_out_robot(net, robot_size=5):
    graph_out = net.activate([1997])
    formated_output = np.reshape(graph_out, (robot_size, robot_size, len(N_TYPES)), "F")
    robot = np.argmax(formated_output, 0)
    for i in range(len(robot)):
        for j in range(len(robot[i])):
            robot[i][j] = robot[i][j]
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

def generate_robot(net, params):
    if params["robot_substrate"] == "CPPN":
        return generate_robot_CPPN_like(net, params["robot_size"])
    else:
        return generate_robot_in_1_out_robot(net, params["robot_size"])

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
