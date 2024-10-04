#!/bin/bash

# Specify the path to the new rosbag file
# BAGFILE_PATH="/home/tdillon/post_processing/tempbag.bag"
# IMAGE_PATH="/home/tdillon/post_processing/image_bank/run_1"
# IMAGE_PATH='/media/tdillon/4D71-BDA7/mapping_data/inkbit_pullback'

# IMAGE_PATH='/home/tdillon/Documents/mapping_data/pvac_dissection_1'

IMAGE_PATH='/home/tdillon/mapping_data/tortuous_dissection_false_lumen_pulsatile_105bpm'

# IMAGE_PATH='/media/tdillon/4D71-BDA7/mapping_data/pvac_dissection_0'
# IMAGE_PATH='/media/tdillon/4D71-BDA7/mapping_data/inkbit_pullback'

# Launch your ROS nodes
roslaunch mapping_ros mapping.launch image_path:=$IMAGE_PATH


