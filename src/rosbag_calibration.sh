#!/bin/bash

# Specify the path to the new rosbag file
# BAGFILE_PATH="/home/tdillon/post_processing/calibration.bag"

# # Create an empty rosbag file
# rosbag record -O $BAGFILE_PATH -a &

# Launch your ROS nodes
# roslaunch calibration_ros calibration.launch bagfile_path:=$BAGFILE_PATH
roslaunch calibration_ros calibration.launch
