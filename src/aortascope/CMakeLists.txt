cmake_minimum_required(VERSION 2.8.3)
project(aortascope)

# Find Python 3.9 interpreter
find_package(Python3 3.9 REQUIRED COMPONENTS Interpreter Development)

# Set the Python executable explicitly
set(PYTHON_EXECUTABLE "/usr/bin/python3.9")



find_package(catkin REQUIRED COMPONENTS
  catkin
  rospy 
  std_msgs 
  geometry_msgs 
  tf
  tf2
  # Add other dependencies as needed
)

catkin_package(CATKIN_DEPENDS 
)


install(PROGRAMS
  src/mapping_parameters_gui.py
  src/mapping_real_time.py
  src/gating_signal_visualizer_white_pump.py
  src/rgb_image_visualizer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

include_directories(${catkin_INCLUDE_DIRS})

catkin_install_python(PROGRAMS
  src/mapping_parameters_gui.py
  src/mapping_real_time.py
  src/gating_signal_visualizer_white_pump.py
  src/rgb_image_visualizer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)


