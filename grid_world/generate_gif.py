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
from grid_world.node import Node
from grid_world.tasks import *
from copy import deepcopy
from PIL import Image, ImageDraw, ImageFont
from mpl_toolkits.axes_grid1 import ImageGrid
import imageio

def generate_sub_image_caption(pil_img, node, grid, best_genome, best_fit):
    draw = ImageDraw.Draw(pil_img)

    # Define the caption text
    caption_text = f"{node.id}"
    font = ImageFont.truetype("notebooks/impact.ttf", size=100)
    caption_size = draw.textsize(caption_text, font)
    caption_position = (10, 10)
    draw.text(caption_position, caption_text, font=font, fill="red")
    
    caption_text = f"{node.task}"
    font = ImageFont.truetype("notebooks/arial.ttf", size=40)
    caption_size = draw.textsize(caption_text, font)
    caption_position = (pil_img.width // 2 - caption_size[0] // 2,15)
    draw.text(caption_position, caption_text, font=font, fill="red")
    

    history = grid.d_historical[best_genome.key]
    caption_text = f"{best_genome.key} ({history.pop_id})"
    font = ImageFont.truetype("notebooks/impact.ttf", size=50)
    caption_size = draw.textsize(caption_text, font)
    caption_position = (pil_img.width - caption_size[0] - 25, 10)
    draw.text(caption_position, caption_text, font=font, fill="green")
    
    
    history = grid.d_historical[best_genome.key]
    caption_text = f"g_id: {best_genome.key}, original_pop: {history.pop_id}, fit: {best_fit}"
    font = ImageFont.truetype("notebooks/arial.ttf", size=40)
    caption_size = draw.textsize(caption_text, font)
    caption_position = (pil_img.width // 2 - caption_size[0] // 2,15+caption_size[1])
    draw.text(caption_position, caption_text, font=font, fill="black")
    
    p1, p2 = history.parent_1, history.parent_2
    if p1 != -1 and p2 != -1:
        caption_text = f"p1: {p1} ({grid.d_historical[p1].pop_id}), p2: {p2} ({grid.d_historical[p2].pop_id})"
    else:
        caption_text = f"First of their name"
    font = ImageFont.truetype("notebooks/arial.ttf", size=40)
    caption_size = draw.textsize(caption_text, font)
    caption_position = (pil_img.width // 2 - caption_size[0] // 2, 30+caption_size[1]*2)
    draw.text(caption_position, caption_text, font=font, fill="black")

def get_robot(genome, neat_config, env, genome_type):
    if genome.robot is not None:
        robot = genome.robot
    else:
        cppn = neat.nn.FeedForwardNetwork.create(genome, neat_config)
        design_substrate = morph_substrate(5, genome_type) # TODO: literal value
        design_net = create_phenotype_network(cppn, design_substrate) 
        robot = generate_robot(design_net, 5) # TODO: literal value
        
    if not eval_robot_constraint(robot):
        return robot, None
    connections = get_full_connectivity(robot)
    
    env = evogym.envs.gym.make(env, body=robot, connections=connections)
    img = env.render(mode='img')
    env.close()
    return robot, img

def generate_grid(grid, n_rows, n_cols, genome_type):
    fig = plt.figure(1,(100,100)) # TODO: literal value
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
        
        robot, img = get_robot(best_genome, value.sgr_pop.neat_config, value.task, genome_type) 
        if img is not None:
            # print(ag.robot)
            pil_img = Image.fromarray(img, 'RGB')
            generate_sub_image_caption(pil_img, value, grid, best_genome, best_genome_fit)
            grid_img[cont].imshow(pil_img,interpolation='none')
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

def main():
    n_rows=6
    n_cols=6
    file_name = "snk_cppn"


    img_arr = []
    for i in range(200, 300, 10):
        print(i)
        with open(f"../island_cp/{file_name}/grid_gen_{i}.pkl", "rb") as file:
            grid = pkl.load(file)
        fig = generate_grid(grid, n_rows, n_cols, grid.params.substrate_type)
        fig.text(0.50, 0.25,  'Generation {i}', horizontalalignment='center', wrap=True, size="xx-large" ) 
        img_arr.append(fig2img(fig))
        plt.clf()


    imageio.mimsave("test" + ".gif", img_arr, duration=(1/2))


if __name__ == "__main__":
    main()