cmake_minimum_required(VERSION 3.0.2)
project(panda_sim_bringup)

# C++ 14
add_compile_options(-std=c++14)

# Warnings
add_definitions(-W -Wall -Wextra
    -Wwrite-strings -Wpointer-arith
    -Winit-self -Wredundant-decls)

# Find catkin macros and libraries
find_package(catkin 
    REQUIRED COMPONENTS
        gazebo_ros
        roscpp
        rospy
        tf
        std_srvs
        std_msgs
)

find_package(gazebo REQUIRED)

catkin_package(
    CATKIN_DEPENDS
)

# Specify additional locations of header files
# Your package locations should be listed before other locations
include_directories(
    include
    ${GAZEBO_INCLUDE_DIRS}
    ${roscpp_INCLUDE_DIRS}
    ${rospy_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
)

link_directories(
    ${GAZEBO_LIBRARY_DIRS}
    ${catkin_LIBRARY_DIRS}
)

# Install all the launch files
install(DIRECTORY launch/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)

# Install all the config files
install(DIRECTORY config/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config)

# Install all the model files
install(DIRECTORY models/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/models)

# Install all the robot files
install(DIRECTORY robots/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/robots)

# Install all the scripts files
install(
  DIRECTORY startup_scripts/
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  USE_SOURCE_PERMISSIONS
)

catkin_python_setup()
