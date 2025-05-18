useful:

``` bash
xhost +
docker exec -it <container-name> bash
```

Firstly install trac_ik:

``` bash
sudo apt update
sudo apt install ros-noetic-trac-ik -y
cd ~/catkin_ws
catkin_make
source devel/setup.bash
cd ~/catkin_ws/src/desired/path
```

Now create a ROS package in the catkin_ws/src folder
copy the "scripts" folder into the created package and make them executable

``` bash
catkin_create_pkg detect_cubes rospy sensor_msgs cv_bridge message_filters tf
```
do this for every file in scripts.
``` bash
chmod +x ~/catkin_ws/src/inverse_kinematics/scripts/file.py
```
or do this but,
> This makes all python files in the directory of the command executable

``` bash
chmod +x *.py
```

Change the path accordingly 
> Tip: change path of your model... /path/to/best.pt

open terminals and follow these steps:
1. roscore
2. roslaunch om_position_controller position_control.launch
3. rosrun detect_cubes perception.py
4. rosrun detect_cubes grab.py