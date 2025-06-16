import numpy as np
import pytransform3d.visualizer as pv
import pytransform3d.trajectories as ptr
from movement_primitives.kinematics import Kinematics
import rosbag
from tf.transformations import quaternion_matrix
from movement_primitives.dmp import CartesianDMP
import pickle
import os
import time
from scipy.interpolate import interp1d
import rospy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tf
import matplotlib.pyplot as plt

def get_cube_position(cube_name, timeout=5.0):
    """Get position of a cube by its TF frame name"""
    print(f"Getting {cube_name} position...")
    if not rospy.core.is_initialized():
        rospy.init_node('tf_xyz_fetcher', anonymous=True)
    
    listener = tf.TransformListener()
    try:
        print(f"Waiting for transform /world -> /{cube_name}...")
        listener.waitForTransform('/world', f'/{cube_name}', rospy.Time(0), rospy.Duration(timeout))
        
        trans, _ = listener.lookupTransform('/world', f'/{cube_name}', rospy.Time(0))
        print(f"{cube_name} position: {trans}")
        return trans
    except Exception as e:
        print(f"Error getting transform for {cube_name}: {e}")
        return None
    
    
CUBE_NAMES = [
    "blue_cube",
    "red_cube",
    "green_cube"
    
]

CUBE_SIZES = [
    0.045,
    0.055,
    0.06
]



def get_cube_stack_state(cube_names, cube_sizes):
    positions = []
    for cube in cube_names:     
        positions.append(get_cube_position(cube))
    positions = np.array(positions)
    min_ind = np.argsort(positions[:,2]) # indices sorted by z
    
    
    A = []
    B = []
    C = []
    stacked = np.zeros(3) - np.ones(3)
    
    for i in range(1,3):
        for j in range (i):
            if positions[min_ind[i], 0] < positions[min_ind[j], 0] + cube_sizes[min_ind[j]] and \
                positions[min_ind[i], 0] > positions[min_ind[j], 0] - cube_sizes[min_ind[j]] and \
                positions[min_ind[i], 1] < positions[min_ind[j], 1] + cube_sizes[min_ind[j]] and \
                positions[min_ind[i], 1] > positions[min_ind[j], 1] - cube_sizes[min_ind[j]]:
                
                stacked[min_ind[i]] = min_ind[j]
    
    

    current_state = np.zeros((np.sum(stacked==-1), 4-np.sum(stacked==-1))) - 1
    print("current state =================")
    print(stacked)
    print(current_state)
    
    for i in range(3):
        if stacked[i] == (-1):
            for j in range(len(current_state)):
                if current_state[j, 0] == -1: 
                    current_state[j, 0] = i
                    k = 1
                    while True: 
                        res = np.argwhere(stacked==i)
                        if len(res)==0:
                            break
                        i = np.squeeze(res[0])
                        current_state[j,k] = i
                        k+=1
                    break
    return current_state + 1
                
                
print(get_cube_stack_state(CUBE_NAMES, CUBE_SIZES))
