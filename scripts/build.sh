#!/usr/bin/env bash

cd sim_ws
catkin build
cd -

cd rob_ws
catkin build
cd -

