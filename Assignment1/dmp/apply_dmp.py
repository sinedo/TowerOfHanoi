#!/usr/bin/env python

import dmp
import sys
import os
import numpy as np
import pytransform3d.trajectories as ptr


import rospy
# Import the specific message type you expect on the topic
# Examples:
# from sensor_msgs.msg import JointState
# from geometry_msgs.msg import PoseStamped
# from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState # <--- CHANGE THIS to your actual message type

# Define the topic name and message type
# You can find these using 'rostopic list' and 'rostopic info /your_topic_name' in the terminal
TOPIC_NAME = "/open_manipulator_6dof/joint_states" # <--- CHANGE THIS to your topic name
MESSAGE_TYPE = JointState    # <--- CHANGE THIS to match the import above

def get_single_message():
    """
    Subscribes to a topic, waits for a single message, and returns it.
    """
    rospy.init_node('dmp_trajectory_publisher', anonymous=True)
    rospy.loginfo(f"Waiting for one message on topic '{TOPIC_NAME}'...")

    try:
        # Wait for a single message from the topic.
        # timeout: Optional duration in seconds to wait. If None, waits indefinitely.
        message = rospy.wait_for_message(TOPIC_NAME, MESSAGE_TYPE, timeout=10.0)

        rospy.loginfo("Message received!")

        # Now you have the 'message' object containing the data
        # Access its fields as needed. For example, if it's JointState:
        if isinstance(message, JointState):
            rospy.loginfo(f"Received JointState message for the starting position with:")
            rospy.loginfo(f"  Header Stamp: {message.header.stamp}")
            rospy.loginfo(f"  Joint Names: {message.name}")
            rospy.loginfo(f"  Positions: {message.position}")
            # rospy.loginfo(f"  Velocities: {message.velocity}") # Often included
            # rospy.loginfo(f"  Efforts: {message.effort}")     # Often included
            # You can return the specific part you need, e.g., positions
            return message
        else:
             # Handle other message types if necessary
             rospy.loginfo(f"Received message data: {message}")
             return message # Return the whole message object

    except rospy.ROSException as e:
        rospy.logerr(f"Failed to get message from '{TOPIC_NAME}' within timeout: {e}")
        return None
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node shutdown while waiting for message.")
        return None


if __name__=="__main__":

    urdf_path = '/root/catkin_ws/src/open_manipulator_friends/open_manipulator_6dof_description/urdf/open_manipulator_6dof.urdf'
    mesh_path = '/root/catkin_ws/src/open_manipulator_friends/open_manipulator_6dof_description/meshes'

    dmp_path = sys.argv[1]

    goal_position_cartesian = np.array(sys.argv[2:5])

    # Initialize the motion generator with the same base_link as dmp_test_1.py
    dmp_gen = dmp.DMPMotionGenerator(
        urdf_path, 
        mesh_path,
        base_link="world"
    )

    if os.path.exists(dmp_path):
        dmp_gen.load_dmp(dmp_path)
    else:
        sys.exit(f"dmp not found\ngenerate DMP first")

    try:
        
        start_position_joint = get_single_message().position
    
    except AttributeError:
        sys.exit(f"Message does not contain position!")

    
    transform_mat = dmp_gen.chain.forward(start_position_joint)
    start_pose_cartesian = ptr.pqs_from_transforms(transform_mat)

    goal_pose_cartesian = dmp_gen.dmp.goal_y.copy()
    goal_pose_cartesian[:3] = goal_position_cartesian

    T, trajectory = dmp_gen.generate_trajectory(start_y=start_pose_cartesian, goal_y=goal_pose_cartesian)

    trajectory, IK_joint_trajectory, gripper_traj ,T = dmp_gen.compute_IK_trajectory(trajectory, dmp_gen.gripper_traj, T ,subsample_factor=10)

    traj_length = min(IK_joint_trajectory.shape[0], gripper_traj.shape[0])

    gripper_traj = gripper_traj[:traj_length]
    IK_joint_trajectory = IK_joint_trajectory[:traj_length,:]
    
    full_trajectory = np.hstack((IK_joint_trajectory, gripper_traj.reshape(-1, 1)))
    # # Interpolate to 20Hz and Save
    interpolated_traj, interpolated_time = dmp.interpolate_joint_trajectory(full_trajectory, T, target_freq=20.0)

    # ROS Publishing
    try:
    
        publisher = dmp.ROSTrajectoryPublisher(['joint1', 'joint2','joint3','joint4','joint5','joint6'])
        publisher.publish_trajectory(interpolated_traj, interpolated_time)
    except rospy.ROSInterruptException:
        print("ROS publishing interrupted.")
   
    










    


    
