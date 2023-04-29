# Robothon 2023

This repository holds the Task Board and "Bring-your-own-device" solution from the S&P Sophia Labs team to the robothon 2023 chalenge. 

## Robotic Platform: Franka Emika Panda Research 3

We have worked on Franka Emika Panda Research 3 (also called FR3 for shorts). FR3 is the newer version of the Panda v1 that we also own and have used extensiely in the past.
The FR3 is a 7 degree of Freedom Co-bot arm with a maximum payload of around 3kg (including end-effector). The FR3 is acapable of to reach a maximum of 2m/s speed at the end-effector level and detect force applied to the end-effector down to 0.8N. We are using the standard Frank Emika Gripper that is a 2-finger gripper with adaptable fingertips

### Hardware Prerequisites

| Component | Description |
| ------ | ------ |
|    Custom fingertips    |    Custom fingertips on the gripper that allow us to grab the probe firmly and enable the routing of the probe cable by alternating through soft and firm grasp    |
|    Custom prob fingertips    |    Custom fingertips on the gripper that allow us to hold the two probs of the multimeter using one robot arm    |
|    Elastic   |    By placing an elastic between the two fingertips we can easily open the door and control the probe cable befoor the first grab     |
|    D455 intel realsense camera   |    RGB sensor allows us to detect the board x and y position and the depth sensor allows to mesure the z position relative to the camera and ultimately relative to the robot arm    |
|    Custom camera mount   |    Keeps the camera in a stable position on the robot arm     |

### Software Prerequisites

- [Python3 and pip3](https://www.python.org/)
- [ROS noetic full desktop](http://wiki.ros.org/noetic/Installation/Ubuntu)
- [Git LFS](https://git-lfs.github.com/)
- [Catkin Tools](https://catkin-tools.readthedocs.io/en/latest/)
- [Realsense Viewer](http://zdome.net/wiki/index.php/Intel_RealSense_Utilities_Install_Ubuntu_20.04_Focal_20210518)
- [yolov7](https://github.com/WongKinYiu/yolov7.git)
- [moveit](https://github.com/ros-planning/moveit.git)
- matplotlib>=3.2.2
- numpy>=1.18.5,<1.24.0
- opencv-python>=4.1.1

## Run the Project

### Install

This repo is using submodules to isolate its modules, no all modules are publicly available on Github so this could prevent you from installing properly if you do not have access to all submodules. Make sure that every repository is correctly pulled before moving to the next steps

DOwanload all submodule with their required commit using ``git submodule update --init`` or by cloning the repo with the ``--recursive`` option
Navigate to root directory and type ``make install``. This will run the install script, installing python requirements specified in requirement.txt 
and run rosdep where needed

### Build

Full build:
``make build``
This essentially runs ``catkin build`` in both workspaces.

Alternatively, if you do not plan on launching the simulation you can build only the robotic workspace (rob_ws):
``cd rob_ws``
``catkin build``
``cd -``

### Run in simulation

You will need to have built both workspaces to make this work.

Source the robotic workspace:
``cd rob_ws``
``source devel/setup.bash``
``cd -``

Do the same for the simulation workspace:
``cd sim_ws``
``source devel/setup.bash``
``cd -``
You can check that your packages have been sourced correctly by testing your env: ``echo $ROS_PACKAGE_PATH``.
You should see sim_ws and rob_ws paths in the output, if one is missing source it again like above

Launch the simulation:
``roslaunch panda_sim_bringup main.launch``
This command should spin up Gazebo and Rviz with Panda setup in an empty world. 
You can start launching external commands in a separate terminal as soon as the green message "ready to plan" appears on your terminal

Launch the tasks:
``roslaunch panda_bringup execute_tasks.launch``
Launching the task will create the scene in Rviz and prompt you to execute a set list of actions, pressing ENTER in your terminal will trigger the next action
You can modify the `rob_ws/src/panda_bringup/launch/execute_tasks.launch` file to point to different tasks files. Additionally you can create or modify task files in `rob_ws/src/panda_bringup/tasks/*.json`. Finally, you can reate or modify scene files in `rob_ws/src/panda_bringup/scenes/*.json`. Note: adding new task or scene files require you to rebuild and source the application before launching again.

### Deploy on FR3/Panda

You do not need to build the simulation workspace for this part.

Source the robot workspace:
``cd rob_ws``
``source devel/setup.bash``
``cd -`

Ensure that your Panda is ready to receive programmatic controls:
- Panda is on
- Panda is not locked
- Panda is activated
- Desk is connected in FCI mode 

Launch the FCI control:
``roslaunch panda_bringup main.launch``
This command should spin up Rviz with the robot mimicking real robot position.
You can start launching external commands when the green message "ready to plan" appears on your terminal

In a separate terminal, source the robot workspace (again):
``cd rob_ws``
``source devel/setup.bash``
``cd -`

Launch the tasks:
``roslaunch panda_bringup execute_tasks.launch``
Launching the task will create the scene in Rviz and prompt you to execute a set list of actions, pressing ENTER in your terminal will unlock the next action
You can modify the `rob_ws/src/panda_bringup/launch/execute_tasks.launch` file to point to different tasks files. Additionally you can create or modify task files in `rob_ws/src/panda_bringup/tasks/*.json`. Finally, you can reate or modify scene files in `rob_ws/src/panda_bringup/scenes/*.json`. Note: adding new task or scene files require you to rebuild and source the application before launching again.

## Architecture

The project is holding 2 catkin workspaces to separate simulation only assets from the main robotic logic. the sim_ws is extending the rob_ws to enable the simulation

The project uses multiple custom modules that can be used independently, most of them work as services

- Image processing: Capture the image(s) and define the configuration of the camera 

- Board detection: Uses image processing, define the logic of the board detection and screen reading (task 2 slider)

- Frame processing: Takes the output of board detection to publish the frame of the detected object in the robot tranformation topic

- Panda Task executor: Implements the logic behind each task, a wrapper to moveit commander and a json interpreter for the task files

- Panda bringup: Holds task files and launch file to start the all project

franka_ros, franka_ros_interface, and libfranka are FR3/Panda specific and are modified for our specific setup and work for all versions of Panda (version can be set in the launch file)

moveit, moveit_msgs, and panda_moveit_config are relative to moveit and its integration on our robot


### Planning Scene and Tasks

- Planning Scene: a static representation of the collisions around your Panda. They can be defined with boxes using a .json format, by default the planning scene is empty

- Basic Tasks: sequential list of actions to execute defined in .json format. Default is a simple joint movemement. Here is the list of available actions:
    - "joint" move joints to the required value. Requires "joint_pose" a list of 7 numbers representing each the joint command for each joint of the panda. (optional) "speed" and "acceleration"
    - "move" move panda end-effector to the required position in the required frame. Requires "x", "y", "z", "rx", "ry", "rz", "rw" and "frame". (optional) "speed" and "acceleration"
    - "gripper" open or closes the gripper to match the required width. Requires "width". (optional) "speed"
    - "grasp" closes the gripper until a certain width or until a certain force is reached. Requires "width", "ei" internal error and "eo" outside error. (optional) "force" and "speed"
    - "angle_constraint" adds a an angle constraints to on joint for the planner. All subsequent tasks will have this constraint until removed. requires "joint" number of the constrainted joint. (optional) "tolerance_above", "tolerance_under" and "priority"
    - "box_constraint" adds a position constraint to a joint for the planner. All subsequent tasks will have this constraint until removed. requires "joint", "x", "y", "z", "size_x", "size_y", "size_z" and "frame". (optional) "visualisation" and "priority"
    - "orientation_constraint", adds an orientation constraint to a joint for the planner. All subsequent tasks will have this constraint until removed. requires ? (optional) "priority"
    - "remove_constraint" clear all existing constraints from the planner
    
- Advanced Tasks
    - "move_in_line" move panda end-effector to the required position in the required frame keeping it in a line between the starting and end position. Warning: this task removes all previous constraint after execution. Requires "x", "y", "z", "rx", "ry", "rz", "rw" and "frame". (optional) "speed" and "acceleration"
    - "spiral search" 
    - "move_until_force" 
    - move_in_line_until_force"

- Specialised Tasks:
    - "slider" mov the panda to a relatively predefined detection pose in the frame and request a detection from "screen_detect", than move to the slider and move the slider acording to the reading
    - "Probe" modification of "spiral search" to work on the task board probing device specifically

For more details, task types are defined in ``/rob_ws/src/panda_task_executor/src/panda_task_executor/task_manager.py``

## Known Issues

- LFS elements missing: it can happen that LFS elements like .png, .stl or .dae files are not imported correctly in the project after clonning it. It can generate all kinds of error when running the code from "file do no exist" to "segmentation faults". If you are experimenting this, make sur git lfs is installed ``sudo apt install git-lfs`` and pull manually the LFS elements with ``git lfs pull`` at the root of the project

