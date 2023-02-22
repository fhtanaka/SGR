import pickle as pkl
import os
from grid_world.graph import Graph
import sys

def main(file_path):
    print(file_path)
    results_dir, test_dir, file_name = file_path.split("/")
    if "pkl" not in file_path:
        print("Not a pkl file")
        return

    try:
        with open(file_path, "rb") as file:
            grid: Graph  = pkl.load(file)
    except:
        print("Couldnt open the pkl")
        return
    img_name = file_name.split(".")[0] 

    n_rows, n_cols = 6, 6
    if "locomotion" in file_path:
        n_rows, n_cols = 4, 4

    grid.save_grid_img(n_rows, n_cols, img_name+ ".jpeg", img_name.replace("_", " "))


if __name__ == "__main__":
    file_path = sys.argv[1]
    # print(file_path)
    main(file_path)
    print()