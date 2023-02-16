import neat
import os
import evogym.envs
from evogym import is_connected, has_actuator, get_full_connectivity, hashable
import numpy as np
import dill
import pickle as pkl
import sys
sys.path.append('../')

from hyperneat.hyperNEAT import create_phenotype_network
from hyperneat.substrate import Substrate
from hyperneat.visualize import draw_net
from dynamic_env.traverser import DynamicObstacleTraverser
from dynamic_env.env_config import EnvConfig


import itertools as it
import time
import imageio

from matplotlib import pyplot as plt


from sgr.substrates import morph_substrate, control_substrate
from sgr.generate_robot import generate_robot, eval_robot_constraint
from sgr.evogym_sim import simulate_env
from sgr.body_speciation import CustomGenome
from grid_world.graph import Graph
from grid_world.node import Node
from grid_world.tasks import *
from copy import deepcopy
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
import imageio


def get_robot(genome, neat_config, env, genome_type):
    if genome.robot is not None:
        robot = genome.robot
    else:
        cppn = neat.nn.FeedForwardNetwork.create(genome, neat_config)
        design_substrate = morph_substrate(5, genome_type)
        design_net = create_phenotype_network(cppn, design_substrate)
        robot = generate_robot(design_net, 5)
        
    if not eval_robot_constraint(robot):
        return robot, None
    connections = get_full_connectivity(robot)
    
    env = evogym.envs.gym.make(env, body=robot, connections=connections)
    img = env.render(mode='img')
    env.close()
    return robot, img

def generate_grid(grid, n_rows, n_cols, genome_type):
    fig = plt.figure(1,(20,20))
    grid_img = ImageGrid(fig, 111,
                    nrows_ncols=(n_rows,n_cols),
                    axes_pad=0.1,
                    share_all=True
                    )

    grid_img[0].get_yaxis().set_ticks([])
    grid_img[0].get_xaxis().set_ticks([])
    cont = 0
    for key, value in grid.d_nodes.items():
        best_genome = None
        best_genome_fit = -100000
        for g in value.sgr_pop.pop.population.values():
            # print(g.key, g.fitness)
            if g.fitness != None and g.fitness > best_genome_fit:
                best_genome = g
                best_genome_fit = g.fitness
        # best_genome = value.sgr_pop.best_genome

        if best_genome == None:
            best_genome = g
        fit = np.round(best_genome_fit, 2)
        task = {value.task}
        # print(f" {best_genome.key} {fit}", end="\t")
        # if cont%n_cols==n_cols-1:
        #     print()
        
        robot, img = get_robot(best_genome, value.sgr_pop.neat_config, value.task, genome_type) 
        if img is not None:
            # print(ag.robot)
            aux = Image.fromarray(img, 'RGB')
            # display(aux.resize((aux.size[0]//3, aux.size[1]//3)))
            grid_img[cont].imshow(aux,interpolation='none')
        cont += 1
    return fig

def fig2img(fig):
    """Convert a Matplotlib figure to a Pillow Image"""
    # Draw the figure on a new surface
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = fig.canvas.tostring_rgb()

    # Convert the buffer to a Pillow Image
    img = Image.frombytes(mode='RGB', size=(w, h), data=buf, decoder_name='raw')
    return img

n_rows=6
n_cols=6
file_name = "snk_cppn"


img_arr = []
for i in range(200, 300, 10):
    print(i)
    with open(f"../island_cp/{file_name}/grid_gen_{i}.pkl", "rb") as file:
        grid: Graph = pkl.load(file)
    fig = generate_grid(grid, n_rows, n_cols, grid.params.substrate_type)
    fig.text(0.5, 0.5, f"Generation {i}", ha="center")
    img_arr.append(fig2img(fig))


imageio.mimsave("test" + ".gif", img_arr, duration=(1/10))