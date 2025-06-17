#!/bin/bash 

while true; do 
    python3 /root/catkin_ws/src/om_position_controller/scripts/detect_state.py > /root/catkin_ws/src/om_position_controller/scripts/current_state.txt
    sleep 1
done
