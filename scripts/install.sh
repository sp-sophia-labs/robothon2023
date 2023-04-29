#!/usr/bin/env bash

set -ex
source /opt/ros/$ROS_DISTRO/setup.sh

# install tools
sudo apt-key adv --keyserver hkps://keyserver.ubuntu.com:443 --recv-key 67170598AF249743
sudo apt install -y wget curl git-lfs unzip python3-pip python3-catkin-tools python3-rosdep

# install python modules
sudo pip3 install -r requirements.txt

# update/install 
git submodule update --init

# install robot dependencies
sudo rosdep init || true
rosdep update
rosdep install --from-paths ./rob_ws --ignore-src -r -y
rosdep install --from-paths ./sim_ws --ignore-src -r -y
