#!/usr/bin/env python3

import rospy
import sys
from trac_ik_python.trac_ik import IK
from std_msgs.msg import Float64
from tf.transformations import quaternion_from_euler
from sensor_msgs.msg import JointState
import xacro
from std_msgs.msg import Header

# IK Documentatino
#http://docs.ros.org/en/melodic/api/trac_ik_python/html/classtrac__ik__python_1_1trac__ik_1_1IK.html

current_pose = None

def update_joint_state(data):
    global current_pose
    current_pose = data.position[:6]

rospy.init_node("ik_jointstate_publisher")

# Subscribe to joint states
rospy.Subscriber("/joint_states", JointState, update_joint_state)
rospy.sleep(1)

while current_pose is None and not rospy.is_shutdown():
    rospy.sleep(0.1)

base = "world"                           # Base Link, needed for IK
endeffector = "end_effector_link"        # End-Effector link, needed for IK

# IK Solver setup
xacro_file = "/root/catkin_ws/src/open_manipulator_friends/open_manipulator_6dof_description/urdf/open_manipulator_6dof.urdf.xacro"
doc = xacro.process_file(xacro_file)
urdf_string = doc.toprettyxml()
ik_solver = IK(base, endeffector, urdf_string=urdf_string, timeout=1, epsilon=1e-3)

# CLI input: x y z roll pitch yaw
x, y, z = map(float, sys.argv[1:4])
roll, pitch, yaw = map(float, sys.argv[4:7])
qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw)
target_pose = [x, y, z, qx, qy, qz, qw]

joint_angles = ik_solver.get_ik(current_pose, *target_pose)

if joint_angles is None:
    rospy.logwarn("No IK solution found.")
    sys.exit(1)

# Create and publish JointState
joint_pub = rospy.Publisher('/gravity_compensation_controller/traj_joint_states', JointState, queue_size=10)
msg = JointState()
msg.header = Header()
msg.header.stamp = rospy.Time.now()
msg.name = [f"joint{i+1}" for i in range(6)] + ["gripper"]
msg.position = list(joint_angles) + [0.0]  # Add dummy gripper value
msg.velocity = [0.0] * 7
msg.effort = [0.0] * 7

# Wait briefly for publisher to register
rospy.sleep(0.5)
joint_pub.publish(msg)
