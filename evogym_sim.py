from evogym import get_full_connectivity
import evogym.envs
import imageio
import numpy as np

def get_obs_size(robot, params):
    connections = get_full_connectivity(robot)
    env = evogym.envs.gym.make(params["env"], body=robot, connections=connections)
    obs = env.reset()
    env.close()
    del env
    return len(obs)

def simulate_env(robot, net, params, render = False, save_gif= False):
    connections = get_full_connectivity(robot)
    env = evogym.envs.gym.make(params["env"], body=robot, connections=connections)

    reward = 0

    obs = env.reset()
    actuators = env.get_actuator_indices("robot")
    finished = False
    imgs = []
    for _ in range(params["steps"]):
        if render:
            env.render('screen')
        elif save_gif:
            imgs.append(env.render(mode='img'))
        
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