import math
from evogym import get_full_connectivity
import evogym.envs
import imageio
import numpy as np

from dynamic_env.traverser import ObstacleTraverser

def get_env(robot, connections, env_name):
    if env_name == "dynamic":
        env = ObstacleTraverser(body=robot, connections=connections)
    else:
        env = evogym.envs.gym.make(env_name, body=robot, connections=connections)
    return env

def get_obs_size(robot, params):
    connections = get_full_connectivity(robot)
    env = get_env(robot, connections, params["env"])
    obs = env.reset()
    env.close()
    del env
    return len(obs)

def simulate_env(robot, net, params, render = False, save_gif= False):
    connections = get_full_connectivity(robot)
    env = get_env(robot, connections, params["env"])
    reward = 0

    obs = env.reset()
    actuators = env.get_actuator_indices("robot")
    in_size = math.ceil(math.sqrt(len(obs))) # this is to be used to format the input

    finished = False
    imgs = []
    for _ in range(params["steps"]):
        if render:
            env.render('screen')
        elif save_gif:
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
    if save_gif:
        imageio.mimsave(params["save_to"] + ".gif", imgs, duration=(1/60))
        return reward, finished
    return reward, finished