#!/usr/bin/env python3

import rospy
import sys
from trac_ik_python.trac_ik import IK
from std_msgs.msg import Float64
from tf.transformations import quaternion_from_euler
from sensor_msgs.msg import JointState
# IK Documentatino
#http://docs.ros.org/en/melodic/api/trac_ik_python/html/classtrac__ik__python_1_1trac__ik_1_1IK.html


current_pose = None

# Subscriber event
def update_joint_state(data):
    global current_pose
    current_pose = data.position[2:8]  #joint data
    #rospy.loginfo(f"current_pose {current_pose}")

base = "world"               # Base Link, needed for IK
endeffector = "end_effector_link"        # End-Effector link, needed for IK

# Initialize the IK solver
# Path is relative, otherwise to long (readability)
urdf = rospy.get_param('/robot_description')
ik_solver = IK(base, endeffector, urdf_string=urdf, timeout=1, epsilon=1e-3)



# Initialize ROS node
rospy.init_node("om_endeffector_position")

# to send data to every servo
publishers = [
    rospy.Publisher(f"/open_manipulator_6dof/joint{i+1}_position/command", Float64, queue_size=10)
    for i in range(6)
]

rospy.Subscriber("/open_manipulator_6dof/joint_states", JointState, update_joint_state)
rospy.sleep(1)  


while current_pose is None and not rospy.is_shutdown():
    rospy.sleep(0.1)

# convert RPY angles to quaternions for endeffector pose
# currently takes in cli for testing 
# TODO: read from subscriber
x, y, z = map(float, sys.argv[1:4])
roll, pitch, yaw = map(float, sys.argv[4:7])
qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw)

# Then pass these to get ik
target_pose = [x, y, z, qx, qy, qz, qw]
# TODO: for loop until subscibed data is read
# start from homeposition and take always the last as first pose

joint_angles = ik_solver.get_ik(current_pose, *target_pose)
# Define the target pose for the end-effector [x, y, z, qx, qy, qz, qw]
#rospy.loginfo(f"target_pose {target_pose}")

# Solve IK to get joint angles for the desired pose
joint_angles = ik_solver.get_ik(current_pose, *target_pose)


if joint_angles is None:
    rospy.logwarn("No Inverse solution found.")
    exit(1)

# Publish the joint angles to move the robot
for i, angle in enumerate(joint_angles):
    publishers[i].publish(angle)
    #rospy.loginfo(f"Joint {i+1}: {angle:.3f}")
