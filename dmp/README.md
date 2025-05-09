save Python files on path catkin_ws/python_files

Start 
- roscore  
- roslaunch om_position_controller roslaunch open_manipulator_6dof_gazebo open_manipulator_6dof_gazebo.launch controller:=position  
- roslaunch open_manipulator_6dof_controller open_manipulator_position_gazebo_controller.launch  

A trajectory has to be learned from the rosbag with the script dmp_learn.py

``` python3 python_files/learn_dmp.py <path_to_rosbag> <name_of_dmp>" ```

To use the dmp, use this command


```python3 python_files/apply_dmp.py <path/to/dmp/dmp.pkl> <end_position x> <end_position y> <end_position z> ```

> **Note:** Not all end-effector positions may be reachable.

```python3 python_files/apply_dmp.py dmp/pick_2.pkl 0.2 0.2 0.05 ```

Return to the start position with the following

```python3 python_files/apply_dmp.py dmp/return.pkl 0.0 0.0 0.3 ```


finally to a new end position
```python3 python_files/apply_dmp.py dmp/pick_2.pkl 0.2 -0.15 0.05```

This is not perfect but it works good. Better teaching trajectories lead to much better outcomes.