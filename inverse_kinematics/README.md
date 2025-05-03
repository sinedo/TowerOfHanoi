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



# Simulation: 
1.) roscore
2.) roslaunch open_manipulator_6dof_gazebo open_manipulator_6dof_gazebo.launch controller:=position
3.) rosrun inverse_kinematics sim.py x y z yaw pitch roll

```
rosrun inverse_kinematics sim.py 0 0 0.2 0 1.5 0
```


# on Hardware
1.) roscore⁠
2.) roslaunch om_position_controller position_control.launch
3.) rosrun inverse_kinematics om.py x y z yaw pitch roll

roslaunch om_position_controller position_control.launch

```
rosrun inverse_kinematics om.py 0 0 0.2 0 1.5 0
```