import math
from evogym import get_full_connectivity
import evogym.envs
import imageio
import numpy as np
import os

from dynamic_env.traverser import DynamicObstacleTraverser

def get_env(robot, connections, env_name):
    if env_name == "dynamic":
        local_dir = os.path.dirname(__file__)
        json_path = os.path.join(local_dir, "../dynamic_env/env.json")
        env = DynamicObstacleTraverser(body=robot, connections=connections, filename=json_path)
    else:
        env = evogym.envs.gym.make(env_name, body=robot, connections=connections)
    return env

def get_obs_size(robot, env_name):
    connections = get_full_connectivity(robot)
    env = get_env(robot, connections, env_name)
    obs = env.reset()
    env.close()
    del env
    return len(obs)

def simulate_env(robot, net, env_name, n_steps, render = False, save_gif=None):
    connections = get_full_connectivity(robot)
    env = get_env(robot, connections, env_name)
    reward = 0

    obs = env.reset()
    actuators = env.get_actuator_indices("robot")
    in_size = math.ceil(math.sqrt(len(obs))) # this is to be used to format the input

    finished = False
    imgs = []
    for _ in range(n_steps):
        if render:
            env.render('screen')
        elif save_gif is not None:
            imgs.append(env.render(mode='img'))
        
        obs.resize(in_size**2, refcheck=False)
        action_by_actuator = net.activate(obs)
        action = np.array([action_by_actuator[i] for i in actuators])

        obs, r, done, _ = env.step(action)
        reward += r

        if done:
            finished = True
            break

    env.close()
    del env
    if save_gif is not None:
        imageio.mimsave(save_gif + ".gif", imgs, duration=(1/60))
        return reward, finished
    return reward, finished