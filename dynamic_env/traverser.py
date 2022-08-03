from evogym.envs.base import EvoGymBase
from gym import error, spaces

from evogym import *
from evogym.envs import WalkingBumpy2, StairsBase

import numpy as np
import os


class DynamicObstacleTraverser(WalkingBumpy2):
    def __init__(self, body, connections=None):

        # make world
        local_dir = os.path.dirname(__file__)
        self.world = EvoWorld.from_json(os.path.join(local_dir, 'data.json'))
        starting_height = (self.world.grid_size[1]//2)+5
        self.world.add_from_array('robot', body, 2, starting_height, connections=connections)

        # init sim
        StairsBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size
        self.sight_dist = 5

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(3 + num_robot_points + (2*self.sight_dist +1),), dtype=np.float)
