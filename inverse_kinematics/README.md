Firstly install trac_ik:

``` bash
sudo apt update
sudo apt install ros-noetic-trac-ik
cd ~/catkin_ws
catkin_make
source
```

Now create a ROS package in the catkin_ws/src folder
copy the "scripts" folder into the created package and make them executable

``` bash
catkin_create_pkg inverse_kinematics rospy moveit_commander geometry_msgs

chmod +x ~/catkin_ws/src/inverse_kinematics/scripts/om.py

chmod +x ~/catkin_ws/src/inverse_kinematics/scripts/sim.py


```


useful:

xhost +
docker exec -it jolly_bhabha bash




1.) roscore
2.) roslaunch open_manipulator_6dof_controller open_manipulator_6dof_controller.launch use_platform:=false ⁠
3.) roslaunch open_manipulator_6dof_gazebo open_manipulator_6dof_gazebo.launch controller:=position

#GUI
4.) roslaunch open_manipulator_6dof_controller open_manipulator_6dof_controller.launch use_platform:=true ⁠
4.) roslaunch open_manipulator_6dof_control_gui open_manipulator_6dof_control_gui.launch

roslaunch om_position_controller position_control.launch