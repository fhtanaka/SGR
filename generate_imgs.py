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
    file_name = file_name.split(".")[0] 

    p1, p2, test_number = file_name.split("_")
    test_number = test_number.zfill(4)
    img_name = f"{p1}_{p2}_{test_number}"

    if "locomotion" in file_path:
        grid.save_grid_img(4,4, img_name+ ".jpeg", img_name.replace("_", " "))
    else:
        grid.save_grid_img(6,6, img_name+ ".jpeg", img_name.replace("_", " "))


if __name__ == "__main__":
    file_path = sys.argv[1]
    # print(file_path)
    main(file_path)
    print()