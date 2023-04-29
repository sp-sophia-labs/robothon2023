cmake_minimum_required(VERSION 3.0.2)
project(panda_bringup)

find_package(catkin REQUIRED COMPONENTS
    geometry_msgs
    roscpp
    rospy
    std_msgs
    tf
    franka_gripper
    moveit_ros_planning_interface
    frame_processing
)

catkin_package(
    CATKIN_DEPENDS
)

link_directories(
    ${catkin_LIBRARY_DIRS}
)

install(PROGRAMS src/panda_bringup/*.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)

install(DIRECTORY tasks/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/tasks) 

include_directories(
    ${roscpp_INCLUDE_DIRS}
    ${rospy_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
)