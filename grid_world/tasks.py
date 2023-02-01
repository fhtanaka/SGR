class Task:
    def __init__(self, name, steps, task_type, difficulty) -> None:
        self.name = name
        self.n_steps = steps
        self.t_type = task_type
        self.difficulty = difficulty

class TaskList:
    def __init__(self) -> None:
        self.task_list = []

        # Walking Tasks
        self.Walker_v0 = Task('Walker-v0', 500, "walking", "easy")
        self.task_list.append(self.Walker_v0)
        self.BridgeWalker_v0 = Task('BridgeWalker-v0', 500, "walking", "medium")
        self.task_list.append(self.BridgeWalker_v0)
        self.BidirectionalWalker_v0 = Task('BidirectionalWalker-v0', 1000, "walking", "medium")
        self.task_list.append(self.BidirectionalWalker_v0)
        # Object Manipulation Tasks
        self.Carrier_v0 = Task('Carrier-v0', 500, "obj_manipualtion", "easy")
        self.task_list.append(self.Carrier_v0)
        self.Pusher_v0 = Task('Pusher-v0', 500, "obj_manipualtion", "easy")
        self.task_list.append(self.Pusher_v0)
        self.BeamToppler_v0 = Task('BeamToppler-v0', 1000, "obj_manipualtion", "easy")
        self.task_list.append(self.BeamToppler_v0)
        self.Pusher_v1 = Task('Pusher-v1', 600, "obj_manipualtion", "medium")
        self.task_list.append(self.Pusher_v1)
        self.Thrower_v0 = Task('Thrower-v0', 300, "obj_manipualtion", "medium")
        self.task_list.append(self.Thrower_v0)
        self.Catcher_v0 = Task('Catcher-v0', 400, "obj_manipualtion", "hard")
        self.task_list.append(self.Catcher_v0)
        self.Carrier_v1 = Task('Carrier-v1', 1000, "obj_manipualtion", "hard")
        self.task_list.append(self.Carrier_v1)
        self.BeamSlider_v0 = Task('BeamSlider-v0', 1000, "obj_manipualtion", "hard")
        self.task_list.append(self.BeamSlider_v0)
        self.Lifter_v0 = Task('Lifter-v0', 300, "obj_manipualtion", "hard")
        self.task_list.append(self.Lifter_v0)
        # Climbing Tasks
        self.Climber_v0 = Task('Climber-v0', 400, "climbing", "medium")
        self.task_list.append(self.Climber_v0)
        self.Climber_v1 = Task('Climber-v1', 600, "climbing", "medium")
        self.task_list.append(self.Climber_v1)
        self.Climber_v2 = Task('Climber-v2', 1000, "climbing", "hard")
        self.task_list.append(self.Climber_v2)
        # Locomotion Tasks
        self.UpStepper_v0 = Task('UpStepper-v0', 600, "locomotion", "medium")
        self.task_list.append(self.UpStepper_v0)
        self.DownStepper_v0 = Task('DownStepper-v0', 500, "locomotion", "easy")
        self.task_list.append(self.DownStepper_v0)
        self.ObstacleTraverser_v0 = Task('ObstacleTraverser-v0', 1000, "locomotion", "medium")
        self.task_list.append(self.ObstacleTraverser_v0)
        self.ObstacleTraverser_v1 = Task('ObstacleTraverser-v1', 1000, "locomotion", "hard")
        self.task_list.append(self.ObstacleTraverser_v1)
        self.Hurdler_v0 = Task('Hurdler-v0', 1000, "locomotion", "hard")
        self.task_list.append(self.Hurdler_v0)
        self.PlatformJumper_v0 = Task('PlatformJumper-v0', 1000, "locomotion", "hard")
        self.task_list.append(self.PlatformJumper_v0)
        self.GapJumper_v0 = Task('GapJumper-v0', 1000, "locomotion", "hard")
        self.task_list.append(self.GapJumper_v0)
        self.Traverser_v0 = Task('Traverser-v0', 1000, "locomotion", "hard")
        self.task_list.append(self.Traverser_v0)
        self.CaveCrawler_v0 = Task('CaveCrawler-v0', 1000, "locomotion", "medium")
        self.task_list.append(self.CaveCrawler_v0)
        # Shape Changer Tasks
        self.AreaMaximizer_v0 = Task('AreaMaximizer-v0', 600, "shape_change", "easy")
        self.task_list.append(self.AreaMaximizer_v0)
        self.AreaMinimizer_v0 = Task('AreaMinimizer-v0', 600, "shape_change", "medium")
        self.task_list.append(self.AreaMinimizer_v0)
        self.WingspanMazimizer_v0 = Task('WingspanMazimizer-v0', 600, "shape_change", "easy")
        self.task_list.append(self.WingspanMazimizer_v0)
        self.HeightMaximizer_v0 = Task('HeightMaximizer-v0', 500, "shape_change", "easy")
        self.task_list.append(self.HeightMaximizer_v0)
        # Misc Tasks
        self.Flipper_v0 = Task('Flipper-v0', 600, "misc", "easy")
        self.task_list.append(self.Flipper_v0)
        self.Jumper_v0 = Task('Jumper-v0', 500, "misc", "easy")
        self.task_list.append(self.Jumper_v0)
        self.Balancer_v0 = Task('Balancer-v0', 600, "misc", "easy")
        self.task_list.append(self.Balancer_v0)
        self.Balancer_v1 = Task('Balancer-v1', 600, "misc", "medium")
        self.task_list.append(self.Balancer_v1)

        self.tasks_by_difficulty = {}
        for t in self.task_list:
            if t.difficulty in self.tasks_by_difficulty:
                self.tasks_by_difficulty[t.difficulty].append(t)
            else:
                self.tasks_by_difficulty[t.difficulty] = [t]

        self.tasks_by_type = {}
        for t in self.task_list:
            if t.t_type in self.tasks_by_type:
                self.tasks_by_type[t.t_type].append(t)
            else:
                self.tasks_by_type[t.t_type] = [t]