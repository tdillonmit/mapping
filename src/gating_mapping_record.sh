#!/bin/bash


IMAGE_PATH='/home/tdillon/mapping_data/gating_test_data/ungated'


# Launch your ROS nodes
roslaunch gating_data_collection gating_data_collection.launch image_path:=$IMAGE_PATH


