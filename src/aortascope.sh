#!/bin/bash

sudo udevadm trigger --subsystem-match=video4linux
udevadm control --reload-rules
udevadm trigger --subsystem-match=video4linux

# Launch your ROS nodes
roslaunch aortascope aortascope.launch


