python grid_main.py -c configs/grid.json --save_to snk_3d_2 --task_grid grid_world/grids/snk.json
python grid_main.py -c configs/grid.json --save_to random_loc_3d_2 --task_grid grid_world/grids/random_loc.json

python grid_main.py -c configs/grid.json --save_to snk_cppn_2 --task_grid grid_world/grids/snk.json --substrate cppn 
python grid_main.py -c configs/grid.json --save_to random_loc_cppn_2 --task_grid grid_world/grids/random_loc.json --substrate cppn 