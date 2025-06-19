#!/bin/bash

for sdf in $(find ../TowerOfHanoi/gazebo_setup | grep sdf$); do
    cp $sdf src/om_position_controller/models/${sdf##*/}
done

[[ -L src/om_position_controller/launch/position_control.launch ]] || {
    cp ../TowerOfHanoi/gazebo_setup/position_control.launch src/om_position_controller/launch/
}

[[ -L final ]] || ln -s ../TowerOfHanoi/final final