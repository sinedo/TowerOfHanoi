Start these on the docker containers:
- roscore  
- roslaunch om_position_controller roslaunch open_manipulator_6dof_gazebo open_manipulator_6dof_gazebo.launch controller:=position  
- roslaunch open_manipulator_6dof_controller open_manipulator_position_gazebo_controller.launch

To use the dmp, use this command, the recorded rosbag is in the argument of the script

```python3 dmp/dmp_motions.py path/to/recordings/rosbag_pick ```

> **Note:** Not all end-effector positions must be reachable.
