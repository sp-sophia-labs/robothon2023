cmake_minimum_required(VERSION 3.0.2)
project(board_detection)

find_package(catkin REQUIRED COMPONENTS
    rospy
)

catkin_python_setup()

catkin_package(
    CATKIN_DEPENDS
)

link_directories(
    ${catkin_LIBRARY_DIRS}
)

install(PROGRAMS src/board_detection/*.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

include_directories(
    ${roscpp_INCLUDE_DIRS}
    ${rospy_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
)