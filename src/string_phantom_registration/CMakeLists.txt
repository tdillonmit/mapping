cmake_minimum_required(VERSION 2.8.3)
project(string_phantom_registration)

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
  src/registration.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

include_directories(${catkin_INCLUDE_DIRS})

catkin_install_python(PROGRAMS
  src/registration.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)


