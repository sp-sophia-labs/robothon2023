#!/usr/bin/env bash

cd sim_ws
catkin build --verbose --mem-limit 60% --pre-clean
cd -

cd rob_ws
catkin build --verbose --mem-limit 60% --pre-clean
cd -