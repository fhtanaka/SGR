python multiple_env_neat.py --substrate cppn -c configs/multiple_env.json --save_to multiple_env_cppn_1 | tee multiple_env_cppn_result_1.txt


python multiple_env_neat.py --substrate cppn -c configs/multiple_env.json --save_to multiple_env_cppn_2 | tee multiple_env_cppn_result_2.txt


python multiple_env_neat.py --substrate cppn -c configs/multiple_env.json --save_to multiple_env_cppn_3 | tee multiple_env_cppn_result_3.txt

python multiple_env_neat.py --substrate 3d -c configs/multiple_env.json --save_to multiple_env_3d_1 | tee multiple_env_3D_result_1.txt
python multiple_env_neat.py --substrate 3d -c configs/multiple_env.json --save_to multiple_env_3d_2 | tee multiple_env_3D_result_2.txt
python multiple_env_neat.py --substrate 3d -c configs/multiple_env.json --save_to multiple_env_3d_3 | tee multiple_env_3D_result_3.txt