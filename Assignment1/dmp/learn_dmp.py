#!/usr/bin/env python

import dmp
import sys
import os




if __name__=="__main__":
    urdf_path = '/root/catkin_ws/src/open_manipulator_friends/open_manipulator_6dof_description/urdf/open_manipulator_6dof.urdf'
    mesh_path = '/root/catkin_ws/src/open_manipulator_friends/open_manipulator_6dof_description/meshes'

    bag_path = sys.argv[1]
    dmp_name = sys.argv[2]

    dmp_path = '/root/catkin_ws/dmp'

    # Initialize the motion generator with the same base_link as dmp_test_1.py
    dmp_gen = dmp.DMPMotionGenerator(
        urdf_path, 
        mesh_path,
        base_link="world"
    )

        # Learn from demonstration
    Y, transforms, joint_traj, gripper_traj = dmp_gen.learn_from_rosbag(
        bag_path, 
        '/gravity_compensation_controller/traj_joint_states'
    )

        
    if os.path.isdir(dmp_path)==0:
        os.mkdir(dmp_path)

    dmp_gen.save_dmp(os.path.join(dmp_path, dmp_name+".pkl"))


