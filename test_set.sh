python grid_main.py -c configs/grid.json --save_to snk_3d_3 --task_grid grid_world/grids/snk.json
./generate_img.sh island_cp_v2 snk_3d_3 2> error.txt 1> output.txt &

python grid_main.py -c configs/grid.json --save_to snk_3d_4 --task_grid grid_world/grids/snk.json
./generate_img.sh island_cp_v2 snk_3d_4 2> error.txt 1> output.txt &

python grid_main.py -c configs/grid.json --save_to snk_3d_5 --task_grid grid_world/grids/snk.json
./generate_img.sh island_cp_v2 snk_3d_5 2> error.txt 1> output.txt &