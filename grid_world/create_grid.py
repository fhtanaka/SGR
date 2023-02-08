import json

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
                value["task"] = task_function(x, y)
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
    create_base_grid()