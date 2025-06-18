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
import geometry_msgs.msg
from geometry_msgs.msg import TransformStamped

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

    nr_cubes = len(cube_names)
    stacked = np.zeros(nr_cubes) - 1

    for i in range(1, nr_cubes):
        for j in range (i):
            if positions[min_ind[i], 0] < positions[min_ind[j], 0] + cube_sizes[min_ind[j]] and \
                positions[min_ind[i], 0] > positions[min_ind[j], 0] - cube_sizes[min_ind[j]] and \
                positions[min_ind[i], 1] < positions[min_ind[j], 1] + cube_sizes[min_ind[j]] and \
                positions[min_ind[i], 1] > positions[min_ind[j], 1] - cube_sizes[min_ind[j]]:

                stacked[min_ind[i]] = min_ind[j]

    current_state = np.zeros((3, nr_cubes)) - 1

    j = 0
    # find all occurences of -1 (cubes on ground)
    for i in range(nr_cubes):
        if stacked[i] == (-1):
            current_state[j, 0] = i
            k = 1
            # fill up with stacked cubes
            while True:
                res = np.argwhere(stacked==i)
                if len(res)==0:
                    break
                i = np.squeeze(res[0])
                current_state[j,k] = i
                k+=1
            j += 1
    result = np.zeros((3, nr_cubes)) - 1

    for i in range(3):
        if current_state[i, 0] == -1:
            continue
        if positions[int(current_state[i, 0]),1] < -0.04:
            result[0] = current_state[i]
        elif -0.04 <= positions[int(current_state[i, 0]),1] <= 0.04:
            result[1] = current_state[i]
        else: # y > 0.04
            result[2] = current_state[i]
    return result + 1


out = ""
for line in get_cube_stack_state(CUBE_NAMES, CUBE_SIZES):
    for value in line:
        out += f"{value},"
    out = f"{out[:-1]}\n"
with open("current_state.txt", "w+") as file:
    file.write(out)
