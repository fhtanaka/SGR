from gym import error, spaces

from evogym import *
from evogym.envs import WalkingBumpy2, StairsBase

import numpy as np
import os

from dynamic_env.generateJSON import generate_env_json
from dynamic_env.env_config import EnvConfig


class DynamicObstacleTraverser(WalkingBumpy2):
    def __init__(self, body, connections=None, filename ="", env_config: EnvConfig=None):

        # make world
        if env_config is not None:
            world_dict = env_config.generate_env_dict()
            self.load_env_from_json_dict(world_dict)
        elif filename != "":
            self.load_world_from_file(filename)
        else:
            world_dict = generate_env_json()
            self.load_env_from_json_dict(world_dict)

        starting_height = (self.world.grid_size[1]//2)+5
        self.world.add_from_array('robot', body, 2, starting_height, connections=connections)

        # init sim
        StairsBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size
        self.sight_dist = 4
        self.step_count = 0

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(3 + num_robot_points + (2*self.sight_dist +1),), dtype=np.float)

    def load_world_from_file(self, filename):
        local_dir = os.path.dirname(__file__)
        self.world = EvoWorld.from_json(os.path.join(local_dir, filename))

    def load_env_from_json_dict(self, world_dict):
        self.world = EvoWorld()
        file_grid_size = Pair(world_dict['grid_width'], world_dict['grid_height'])

        for name, obj_data in world_dict['objects'].items():
            obj = WorldObject()
            obj.load_from_parsed_json(name, obj_data, file_grid_size)
            self.world.add_object(obj) 

    def get_obs(self):
        obs = np.array ([
            *self.get_vel_com_obs("robot"),
            *self.get_ort_obs("robot"),
            *self.get_pos_com_obs("robot"),
            *self.get_floor_obs("robot", ["ground"], self.sight_dist),
            self.step_count%30,
        ])
        return obs

    def step(self, action):
        _, reward, done, _ =  super().step(action)
        obs = self.get_obs()
        return obs, reward, done, {}

    def reset(self):
        
        _ = super().reset()
        return self.get_obs()