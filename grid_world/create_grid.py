import json
from tasks import Task
import numpy as np
def locomotion_task():
    def task_func(x, y):
        aux = y%4
        if aux == 1:
            return "UpStepper-v0"
        elif aux == 2:
            return "Hurdler-v0"
        return "Walker-v0"

    create_base_grid(file_name="./locomotion.json", task_function=task_func)

def manipulation_task():
    def task_func(x, y):
        aux = y%4
        if aux == 1:
            return "Thrower-v0"
        elif aux == 2:
            return "Carrier-v1"
        return "Carrier-v0"

    create_base_grid(file_name="./manipulation.json", task_function=task_func)

def snk_task():
    def task_func(x, y):
        if x == 1 and y != 0 and y != 5:
            return "ObstacleTraverser-v1"
        elif x == 4 and y != 0 and y != 5:
            return "ObstacleTraverser-v1"
        elif y == 1 and x != 0 and x != 5:
            return "ObstacleTraverser-v1"
        elif y == 4 and x != 0 and x != 5:
            return "ObstacleTraverser-v1"
        return "ObstacleTraverser-v0"
    create_base_grid(size=(6,6), file_name="./snk.json", task_function=task_func)

def random_locomotion_task():

    possible_tasks = [
        "Walker-v0",
        "BridgeWalker-v0",
        "UpStepper-v0",
        "DownStepper-v0",
        "ObstacleTraverser-v0",
        "ObstacleTraverser-v1",
        "Hurdler-v0",
        "PlatformJumper-v0",
        "GapJumper-v0",
    ]
    def task_func(x, y):
        task =  np.random.choice(possible_tasks, 1)[0]
        print(task)
        return task
    create_base_grid(size=(6,6), file_name="./random_loc.json", task_function=task_func)

def create_base_grid(size=(4,4), file_name="./temp.json", task_function=None):
    x, y = size

    grid = []
    for i in range(x):
        grid.append([])
        for j in range(y):
            grid[i].append(j + i*y)

    for i in grid:
        print(i)

    json_grid = {}
    for i in range(x):
        for j in range(y):
            value = {
                "neighbors":[]
            }
            if task_function != None:
                value["task"] = task_function(i, j)
                print(i, j, value["task"])
            else:
                value["task"] = "Walker-v0"

            for i2 in range(i-1, i+2):
                for j2 in range(j-1, j+2):
                    if i2 < 0 or i2 >= x or j2 < 0 or j2 >= y or grid[i2][j2] == grid[i][j]:
                        continue
                    value["neighbors"].append(grid[i2][j2])

            json_grid[grid[i][j]] = value
    
    with open(file_name, 'w') as f:
        json.dump(json_grid, f, indent = 4, ensure_ascii = False)

if __name__ == "__main__":
    random_locomotion_task()