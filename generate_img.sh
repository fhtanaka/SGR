#!/bin/bash

# declare -a arr=("locomotion_3d" "locomotion_cppn" "snk_3d" "snk_cppn" "random_loc_3d" "random_loc_cppn" "snk_3d_2" "snk_cppn_2" "random_loc_3d_2" "random_loc_cppn_2")

# Set the directory path
dir_path="$1"

# Loop over the files in the directory
for dir in "${@:2}"
do
    for file in "${dir_path}/${dir}"/*.pkl
    do
    # Check if the file is a regular file
    if [[ -f "$file" ]]; then
        # Call the Python function with the file as an argument
        python generate_imgs.py "${file}"
    fi
    done
done