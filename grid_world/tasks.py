from typing import List, Dict
import gym
import numpy as np
import evogym.envs
class Task:
    def __init__(self, name, steps, task_type, difficulty) -> None:
        self.name = name
        self.n_steps = steps
        self.t_type = task_type
        self.difficulty = difficulty

class TaskList:
    def __init__(self) -> None:
        self.task_dict: Dict[str, Task] = {}

        # Walking Tasks
        self.Walker_v0 = Task('Walker-v0', 600, "walking", "easy")
        self.task_dict["Walker-v0"]=self.Walker_v0
        self.BridgeWalker_v0 = Task('BridgeWalker-v0', 600, "walking", "medium")
        self.task_dict["BridgeWalker-v0"]=self.BridgeWalker_v0
        self.BidirectionalWalker_v0 = Task('BidirectionalWalker-v0', 800, "walking", "medium")
        self.task_dict["BidirectionalWalker-v0"]=self.BidirectionalWalker_v0
        # Object Manipulation Tasks
        self.Carrier_v0 = Task('Carrier-v0', 600, "obj_manipualtion", "easy")
        self.task_dict["Carrier-v0"]=self.Carrier_v0
        self.Pusher_v0 = Task('Pusher-v0', 600, "obj_manipualtion", "easy")
        self.task_dict["Pusher-v0"]=self.Pusher_v0
        self.BeamToppler_v0 = Task('BeamToppler-v0', 800, "obj_manipualtion", "easy")
        self.task_dict["BeamToppler-v0"]=self.BeamToppler_v0
        self.Pusher_v1 = Task('Pusher-v1', 600, "obj_manipualtion", "medium")
        self.task_dict["Pusher-v1"]=self.Pusher_v1
        self.Thrower_v0 = Task('Thrower-v0', 300, "obj_manipualtion", "medium")
        self.task_dict["Thrower-v0"]=self.Thrower_v0
        self.Catcher_v0 = Task('Catcher-v0', 400, "obj_manipualtion", "hard")
        self.task_dict["Catcher-v0"]=self.Catcher_v0
        self.Carrier_v1 = Task('Carrier-v1', 800, "obj_manipualtion", "hard")
        self.task_dict["Carrier-v1"]=self.Carrier_v1
        self.BeamSlider_v0 = Task('BeamSlider-v0', 800, "obj_manipualtion", "hard")
        self.task_dict["BeamSlider-v0"]=self.BeamSlider_v0
        self.Lifter_v0 = Task('Lifter-v0', 300, "obj_manipualtion", "hard")
        self.task_dict["Lifter-v0"]=self.Lifter_v0
        # Climbing Tasks
        self.Climber_v0 = Task('Climber-v0', 400, "climbing", "medium")
        self.task_dict["Climber-v0"]=self.Climber_v0
        self.Climber_v1 = Task('Climber-v1', 600, "climbing", "medium")
        self.task_dict["Climber-v1"]=self.Climber_v1
        self.Climber_v2 = Task('Climber-v2', 800, "climbing", "hard")
        self.task_dict["Climber-v2"]=self.Climber_v2
        # Locomotion Tasks
        self.UpStepper_v0 = Task('UpStepper-v0', 600, "locomotion", "medium")
        self.task_dict["UpStepper-v0"]=self.UpStepper_v0
        self.DownStepper_v0 = Task('DownStepper-v0', 600, "locomotion", "easy")
        self.task_dict["DownStepper-v0"]=self.DownStepper_v0
        self.ObstacleTraverser_v0 = Task('ObstacleTraverser-v0', 800, "locomotion", "medium")
        self.task_dict["ObstacleTraverser-v0"]=self.ObstacleTraverser_v0
        self.ObstacleTraverser_v1 = Task('ObstacleTraverser-v1', 800, "locomotion", "hard")
        self.task_dict["ObstacleTraverser-v1"]=self.ObstacleTraverser_v1
        self.Hurdler_v0 = Task('Hurdler-v0', 800, "locomotion", "hard")
        self.task_dict["Hurdler-v0"]=self.Hurdler_v0
        self.PlatformJumper_v0 = Task('PlatformJumper-v0', 800, "locomotion", "hard")
        self.task_dict["PlatformJumper-v0"]=self.PlatformJumper_v0
        self.GapJumper_v0 = Task('GapJumper-v0', 800, "locomotion", "hard")
        self.task_dict["GapJumper-v0"]=self.GapJumper_v0
        self.Traverser_v0 = Task('Traverser-v0', 800, "locomotion", "hard")
        self.task_dict["Traverser-v0"]=self.Traverser_v0
        self.CaveCrawler_v0 = Task('CaveCrawler-v0', 800, "locomotion", "medium")
        self.task_dict["CaveCrawler-v0"]=self.CaveCrawler_v0
        # Shape Changer Tasks
        self.AreaMaximizer_v0 = Task('AreaMaximizer-v0', 600, "shape_change", "easy")
        self.task_dict["AreaMaximizer-v0"]=self.AreaMaximizer_v0
        self.AreaMinimizer_v0 = Task('AreaMinimizer-v0', 600, "shape_change", "medium")
        self.task_dict["AreaMinimizer-v0"]=self.AreaMinimizer_v0
        self.WingspanMazimizer_v0 = Task('WingspanMazimizer-v0', 600, "shape_change", "easy")
        self.task_dict["WingspanMazimizer-v0"]=self.WingspanMazimizer_v0
        self.HeightMaximizer_v0 = Task('HeightMaximizer-v0', 600, "shape_change", "easy")
        self.task_dict["HeightMaximizer-v0"]=self.HeightMaximizer_v0
        # Misc Tasks
        self.Flipper_v0 = Task('Flipper-v0', 600, "misc", "easy")
        self.task_dict["Flipper-v0"]=self.Flipper_v0
        self.Jumper_v0 = Task('Jumper-v0', 600, "misc", "easy")
        self.task_dict["Jumper-v0"]=self.Jumper_v0
        self.Balancer_v0 = Task('Balancer-v0', 600, "misc", "easy")
        self.task_dict["Balancer-v0"]=self.Balancer_v0
        self.Balancer_v1 = Task('Balancer-v1', 600, "misc", "medium")
        self.task_dict["Balancer-v1"]=self.Balancer_v1

        self.tasks_by_difficulty: Dict[str, List[Task]] = {}
        for t in self.task_dict.values():
            if t.difficulty in self.tasks_by_difficulty:
                self.tasks_by_difficulty[t.difficulty].append(t)
            else:
                self.tasks_by_difficulty[t.difficulty] = [t]

        self.tasks_by_type: Dict[str, List[Task]] = {}
        for t in self.task_dict.values():
            if t.t_type in self.tasks_by_type:
                self.tasks_by_type[t.t_type].append(t)
            else:
                self.tasks_by_type[t.t_type] = [t]

def get_locomotion_env_obs(env: evogym.envs.WalkingFlat):
    obs = np.array ([
        *env.get_pos_com_obs("robot"),
        *env.get_vel_com_obs("robot"),
        *env.get_ort_obs("robot"),
        *env.get_floor_obs("robot", ["ground"], 5),
    ])
    return obs

def get_manipulation_env_obs(env: evogym.envs.CarrySmallRect):
    obs = np.array ([
        *env.get_pos_com_obs("robot"),
        *env.get_vel_com_obs("robot"),
        *env.get_ort_obs("robot"),
        *env.get_floor_obs("robot", ["ground"], 5),
    ])
    return obs