#!/bin/bash

declare -a arr=("locomotion_cppn" "snk_cppn" "random_loc_cppn" "snk_3d" "random_loc_3d" )

# Set the directory path
dir_path="island_cp/"

# Loop over the files in the directory
for dir in "${arr[@]}"
do
    for file in "${dir_path}${dir}"/*
    do
    # Check if the file is a regular file
    if [[ -f "$file" ]]; then
        # Call the Python function with the file as an argument
        python generate_imgs.py "${file}"
    fi
    done
done