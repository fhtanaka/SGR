"""
All Hyperneat related logic resides here.
"""

import neat


def create_phenotype_network(cppn, substrate, activation_function="tanh", output_activation="identity", output_node_idx=0):
    """
    Creates a recurrent network using a cppn and a substrate.
    """
    input_coordinates = substrate.input_coordinates
    output_coordinates = substrate.output_coordinates
    hidden_coordinates = substrate.hidden_coordinates
    # Get activation function.
    act_function_set = neat.activations.ActivationFunctionSet()
    activation = act_function_set.get(activation_function)
    out_activation = act_function_set.get(output_activation)

    idx = 0
    node_dict = {}
    for n in input_coordinates:
        node_dict[n] = idx
        idx += 1
    for layer in hidden_coordinates:
        for n in layer:
            node_dict[n] = idx
            idx += 1
    for n in output_coordinates:
        node_dict[n] = idx
        idx += 1

    

    node_evals = []
    # connect input to hidden
    if len(hidden_coordinates) > 0:
        for node in hidden_coordinates[0]:
            im = connect_node_to_layer(cppn, node, input_coordinates, node_dict, False, 1, output_node_idx)
            eval = (node_dict[node], activation, sum, 0.0, 1.0, im)
            node_evals.append(eval)
    # connect input to output if there are no hidden layers
    else:
        for node in output_coordinates:
            im = connect_node_to_layer(cppn, node, input_coordinates, node_dict, False, 1, output_node_idx)
            eval = (node_dict[node], out_activation, sum, 0.0, 1.0, im)
            node_evals.append(eval)

    # connect hidden to hidden
    l = 0
    while l+1 < len(hidden_coordinates):
        for node in hidden_coordinates[l+1]:
            im = connect_node_to_layer(cppn, node, hidden_coordinates[l], node_dict, False, 1, output_node_idx)
            eval = (node_dict[node], activation, sum, 0.0, 1.0, im)
            node_evals.append(eval)
        l += 1

    # connect hidden to output
    if len(hidden_coordinates) > 0:
        for node in output_coordinates:
            im = connect_node_to_layer(cppn, node, hidden_coordinates[-1], node_dict, False, 1, output_node_idx)
            eval = (node_dict[node], out_activation, sum, 0.0, 1.0, im)
            node_evals.append(eval)


    input_nodes = [node_dict[n] for n in input_coordinates]
    output_nodes = [node_dict[n] for n in output_coordinates]
    return neat.nn.FeedForwardNetwork(input_nodes, output_nodes, node_evals)

def connect_node_to_layer(cppn, n_coord, goal_layer, node_dict, outgoing, max_weight, output_node_idx):
    im = []
    for node in goal_layer:
        w = query_cppn(n_coord, node, outgoing, cppn, max_weight, output_node_idx)
        if w != 0.0:  # Only include connection if the weight isn't 0.0.
            im.append((node_dict[node], w))
    return im

def query_cppn(coord1, coord2, outgoing, cppn, max_weight, output_node_idx):
    """
    Get the weight from one point to another using the CPPN.
    Takes into consideration which point is source/target.
    """

    if outgoing:
        i = [*coord1, *coord2, 1.0]
    else:
        i = [*coord2, *coord1, 1.0]
    w = cppn.activate(i)[output_node_idx]
    if abs(w) > 0.2:  # If abs(weight) is below threshold, treat weight as 0.0.
        if w > 0:
            w = (w - 0.2) / 0.8
        else:
            w = (w + 0.2) / 0.8
        return w * max_weight
    else:
        return 0.0