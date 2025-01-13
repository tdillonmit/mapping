#!/bin/bash

# Specify the path to the new rosbag file
# BAGFILE_PATH="/home/tdillon/post_processing/tempbag.bag"

IMAGE_PATH='None'

# Launch your ROS nodes
roslaunch mapping_ros mapping.launch image_path:=$IMAGE_PATH
