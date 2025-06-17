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
from scipy.spatial.transform import Rotation
import rospy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tf
import matplotlib.pyplot as plt
import itertools

from tf.transformations import euler_from_matrix, euler_matrix

class DMPMotionGenerator:
    def __init__(self, urdf_path, mesh_path=None, joint_names=None, base_link="world", end_effector_link="end_effector_link"):
        print("Initializing DMPMotionGenerator for Gazebo...")
        self.urdf_path = urdf_path
        self.mesh_path = mesh_path
        self.kin = self._load_kinematics(urdf_path, mesh_path)
        
        self.joint_names = joint_names or ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.gripper_joint_names = ["gripper", "gripper_sub"]
        self.base_link = base_link
        self.end_effector_link = end_effector_link
        self.chain = self.kin.create_chain(self.joint_names, base_link, end_effector_link)
        self.dmp = None
        self.IK_joint_trajectory = None
        self.gripper_trajectory = None
        
        if not rospy.core.is_initialized():
            rospy.init_node('dmp_motion_generator', anonymous=True)

    def _load_kinematics(self, urdf_path, mesh_path=None):
        with open(urdf_path, 'r') as f:
            return Kinematics(f.read(), mesh_path=mesh_path)

    def learn_from_rosbag(self, bag_path, joint_topic, dt=None, n_weights=10):
        transforms, joint_trajectory, gripper_trajectory, time_stamp = self._process_rosbag(bag_path, joint_topic)
        self.gripper_trajectory = gripper_trajectory
        
        print(f"Transforms shape: {transforms.shape}")
        Y = ptr.pqs_from_transforms(transforms[10:,:,:])
        if dt is None:
            dt = 1/self.frequency
        self.dmp = CartesianDMP(execution_time=max(time_stamp), dt=dt, n_weights_per_dim=n_weights)
        self.dmp.imitate(time_stamp[10:], Y)
        
        return Y, transforms, joint_trajectory, gripper_trajectory

    def _process_rosbag(self, bag_path, joint_topic):
        transforms = []
        joint_trajectory = []
        gripper_trajectory = []
        time_stamp = []
        
        print(f"Reading bag file: {bag_path}")
        bag = rosbag.Bag(bag_path)
        for topic, msg, t in bag.read_messages(topics=[joint_topic]):
            joint_pos = msg.position[:6]
            gripper_pos = msg.position[6] if len(msg.position) > 6 else 0.0
            joint_trajectory.append(joint_pos)
            gripper_trajectory.append(gripper_pos)

            transforms.append(self.chain.forward(joint_pos))
            time_stamp.append(msg.header.stamp.to_sec())    
        bag.close()
        
        transforms = np.array(transforms)
        joint_trajectory = np.array(joint_trajectory)
        gripper_trajectory = np.array(gripper_trajectory)
        time_stamp = np.array(time_stamp)
        
        dt = []
        for i in range(1, time_stamp.shape[0]):
            dt.append(time_stamp[i]- time_stamp[i-1])
        self.frequency = 1/ np.average(np.array(dt))
        
        positions = np.array([T[:3, 3] for T in transforms])
        mask, _ = self.remove_outliers_mad(positions, threshold=5.0)
        
        filtered_time = time_stamp[mask]
        normalized_time = filtered_time - filtered_time[0]
        
        return transforms[mask], joint_trajectory[mask], gripper_trajectory[mask], normalized_time

    def remove_outliers_mad(self, data, threshold=3.5):
        median = np.median(data, axis=0)
        diff = np.abs(data - median)
        mad = np.median(diff, axis=0)
        modified_z_score = 0.6745 * diff / (mad + 1e-6)
        mask = np.all(modified_z_score < threshold, axis=1)
        return mask, data[mask]

    def generate_trajectory(self, start_y=None, goal_y=None):
        print(f"Generating trajectory")
        if self.dmp is None:
            raise ValueError("No DMP model available. Learn or load a model first.")
            
        if start_y is not None:
            self.dmp.start_y = start_y
            print(f"Using custom start: {start_y}")
        else:
            print(f"Using default start: {self.dmp.start_y}")
            
        if goal_y is not None:
            self.dmp.goal_y = goal_y
            print(f"Using custom goal: {goal_y}")
        else:
            print(f"Using default goal: {self.dmp.goal_y}")
        
        T, Y = self.dmp.open_loop()
        trajectory = ptr.transforms_from_pqs(Y)
        return T, trajectory

    def save_dmp(self, filepath):
        if self.dmp is None:
            rospy.logerr("No DMP model available to save.")
            return
        if self.gripper_trajectory is None:
            rospy.logwarn("Gripper trajectory not available or not learned. Saving None for gripper_trajectory.")

        data_to_save = {
            'dmp': self.dmp,
            'gripper_trajectory': self.gripper_trajectory
        }
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data_to_save, f)
            rospy.loginfo(f"DMP and gripper trajectory saved to {filepath}")
        except Exception as e:
            rospy.logerr(f"Failed to save DMP data to {filepath}: {e}")

    def load_dmp(self, filepath):
        rospy.loginfo(f"Loading DMP data from {filepath}")
        try:
            with open(filepath, 'rb') as f:
                loaded_data = pickle.load(f)

            if isinstance(loaded_data, dict):
                if 'dmp' in loaded_data:
                    self.dmp = loaded_data['dmp']
                else:
                    rospy.logerr("Loaded dictionary is missing 'dmp' key.")
                    self.dmp = None

                if 'gripper_trajectory' in loaded_data:
                    self.gripper_trajectory = loaded_data['gripper_trajectory']
                    if self.gripper_trajectory is not None:
                         rospy.loginfo(f"Gripper trajectory loaded ({len(self.gripper_trajectory)} points).")
                    else:
                         rospy.loginfo("Loaded None for gripper trajectory.")
                else:
                    rospy.logwarn("Loaded dictionary is missing 'gripper_trajectory' key. Setting to None.")
                    self.gripper_trajectory = None
            else:
                rospy.logwarn("Loading old DMP format (only DMP object found). Gripper trajectory will be None.")
                self.dmp = loaded_data
                self.gripper_trajectory = None

            if self.dmp:
                rospy.loginfo("DMP object loaded successfully.")
            else:
                 rospy.logerr("Failed to load DMP object.")

        except FileNotFoundError:
            rospy.logerr(f"DMP file not found: {filepath}")
            self.dmp = None
            self.gripper_trajectory = None
        except Exception as e:
            rospy.logerr(f"Error loading DMP data from {filepath}: {e}")
            self.dmp = None
            self.gripper_trajectory = None
    
    def compute_IK_trajectory(self, trajectory, time_stamp, starting_joint_config = None, q0=None, subsample_factor=1):
        if q0 is None:
            q0 = np.array([-0.03834952, -0.84062147, 1.26093221, 0.00613592, 1.97576725, -0.00460194])
        
        if subsample_factor > 1:
            subsampled_trajectory = trajectory[::subsample_factor]
            subsampled_time_stamp = time_stamp[::subsample_factor]
            subsampled_gripper_trajectory = self.gripper_trajectory[::subsample_factor] if self.gripper_trajectory is not None else None
            print(f"Subsampled time from {len(time_stamp)} to {len(subsampled_time_stamp)} points")
            print(f"Subsampled trajectory from {len(trajectory)} to {len(subsampled_trajectory)} points")
        else:
            subsampled_trajectory = trajectory
            subsampled_time_stamp = time_stamp
            subsampled_gripper_trajectory = self.gripper_trajectory
        
        print(f"Solving inverse kinematics for {len(subsampled_trajectory)} points...")
        
        start_time = time.time()
        
        random_state = np.random.RandomState(0)
        joint_trajectory = self.chain.inverse_trajectory(
            subsampled_trajectory, random_state=random_state, orientation_weight=1.0)
            
        print(f"IK solved in {time.time() - start_time:.2f} seconds")
        
        return subsampled_trajectory, joint_trajectory, subsampled_gripper_trajectory, subsampled_time_stamp


    def visualize_trajectory(self, trajectory, joint_trajectory, q0=None):
        print(f"Plotting trajectory...")
        fig = pv.figure()
        fig.plot_transform(s=0.3)
        
        graph = fig.plot_graph(
            self.kin.tm, "world", show_visuals=False, show_collision_objects=True,
            show_frames=True, s=0.1, whitelist=[self.base_link, self.end_effector_link])

        fig.plot_transform(trajectory[0], s=0.15)
        fig.plot_transform(trajectory[-1], s=0.15)
        
        pv.Trajectory(trajectory, s=0.05).add_artist(fig)
        
        fig.view_init()
        fig.animate(
            animation_callback, len(trajectory), loop=True,
            fargs=(graph, self.chain, joint_trajectory))
        fig.show()


class GazeboTrajectoryPublisher:
    def __init__(self, joint_names=None, gripper_joint_names=None):
        if not rospy.core.is_initialized():
            rospy.init_node("gazebo_trajectory_publisher", anonymous=True)
        
        self.joint_names = joint_names or ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.gripper_joint_names = gripper_joint_names or ["gripper", "gripper_sub"]
        
        self.arm_pub = rospy.Publisher('/open_manipulator_6dof/arm_controller/command', 
                                     JointTrajectory, queue_size=10)
        self.gripper_pub = rospy.Publisher('/open_manipulator_6dof/gripper_controller/command', 
                                         JointTrajectory, queue_size=10)
        
        print(f"[Gazebo] Initialized publishers:")
        print(f"  - Arm: /open_manipulator_6dof/arm_controller/command")
        print(f"  - Gripper: /open_manipulator_6dof/gripper_controller/command")
        
        rospy.sleep(1.0)

    def publish_trajectory(self, joint_trajectory, gripper_trajectory, timestamps, execute_time_factor=1.0):
        if len(joint_trajectory) == 0:
            rospy.logwarn("[Gazebo] Empty trajectory provided")
            return
        
        print(f"[Gazebo] Publishing trajectory with {len(joint_trajectory)} points")
        
        arm_msg = JointTrajectory()
        arm_msg.header.stamp = rospy.Time.now()
        arm_msg.joint_names = self.joint_names
        
        gripper_msg = JointTrajectory()
        gripper_msg.header.stamp = rospy.Time.now()
        gripper_msg.joint_names = self.gripper_joint_names
        
        for i in range(len(joint_trajectory)):
            arm_point = JointTrajectoryPoint()
            arm_point.positions = joint_trajectory[i].tolist()
            arm_point.velocities = [0.0] * len(self.joint_names)
            arm_point.accelerations = [0.0] * len(self.joint_names)
            arm_point.time_from_start = rospy.Duration.from_sec(
                (timestamps[i] - timestamps[0]) * execute_time_factor
            )
            arm_msg.points.append(arm_point)
            
            if gripper_trajectory is not None and i < len(gripper_trajectory):
                gripper_point = JointTrajectoryPoint()
                gripper_value = gripper_trajectory[i]
                gripper_point.positions = [-2.0*gripper_value, -2.0*gripper_value]
                gripper_point.velocities = [0.0, 0.0]
                gripper_point.accelerations = [0.0, 0.0]
                gripper_point.time_from_start = rospy.Duration.from_sec(
                    (timestamps[i] - timestamps[0]) * execute_time_factor
                )
                gripper_msg.points.append(gripper_point)
        
        print(f"[Gazebo] Publishing arm trajectory with {len(arm_msg.points)} points")
        self.arm_pub.publish(arm_msg)
        
        if gripper_trajectory is not None and len(gripper_msg.points) > 0:
            print(f"[Gazebo] Publishing gripper trajectory with {len(gripper_msg.points)} points")
            self.gripper_pub.publish(gripper_msg)
        else:
            print(f"[Gazebo] No gripper trajectory to publish")
        
        print(f"[Gazebo] Trajectory published successfully")

    def publish_single_trajectory(self, full_trajectory, timestamps, execute_time_factor=1.0):
        if full_trajectory.shape[1] >= 6:
            arm_traj = full_trajectory[:, :6]
            gripper_traj = full_trajectory[:, 6] if full_trajectory.shape[1] > 6 else None
            
            self.publish_trajectory(arm_traj, gripper_traj, timestamps, execute_time_factor)
        else:
            rospy.logwarn(f"[Gazebo] Invalid trajectory shape: {full_trajectory.shape}")

    def publish_home_position(self, home_position=None, execution_time=5.0):
        if home_position is None:
            home_position = [-0.03834952, -0.84062147, 1.26093221, 0.00613592, 1.97576725, -0.00460194]
        
        print(f"[Gazebo] Publishing home position command...")
        print(f"[Gazebo] Home position: {home_position}")
        print(f"[Gazebo] Execution time: {execution_time} seconds")
        
        arm_msg = JointTrajectory()
        arm_msg.header.stamp = rospy.Time.now()
        arm_msg.joint_names = self.joint_names
        
        home_point = JointTrajectoryPoint()
        home_point.positions = home_position
        home_point.velocities = [0.0] * len(self.joint_names)
        home_point.accelerations = [0.0] * len(self.joint_names)
        home_point.time_from_start = rospy.Duration.from_sec(execution_time)
        
        arm_msg.points.append(home_point)
        
        self.arm_pub.publish(arm_msg)
        print(f"[Gazebo] Home position command published and latched")


def animation_callback(step, graph, chain, joint_trajectory):
    chain.forward(joint_trajectory[step])
    graph.set_data()
    return graph

def save_trajectory_data(joint_trajectory, timestamps, filepath):
    data = {
        'trajectory': joint_trajectory,
        'timestamps': timestamps
    }
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"[SAVE] Trajectory data saved to {filepath}")

def load_trajectory_data(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    joint_trajectory = data['trajectory']
    timestamps = data['timestamps']
    print(f"[LOAD] Loaded trajectory from {filepath} (length={len(joint_trajectory)})")
    return joint_trajectory, timestamps

def interpolate_joint_trajectory(joint_traj, time_stamps, target_freq=20.0):
    num_joints = joint_traj.shape[1]
    duration = time_stamps[-1] - time_stamps[0]
    num_samples = int(duration * target_freq)
    new_timestamps = np.linspace(time_stamps[0], time_stamps[-1], num_samples)
    
    interp_traj = np.zeros((num_samples, num_joints))
    for i in range(num_joints):
        interpolator = interp1d(time_stamps, joint_traj[:, i], kind='linear', fill_value="extrapolate")
        interp_traj[:, i] = interpolator(new_timestamps)
    
    return interp_traj, new_timestamps

def get_cube_position(cube_name, timeout=5.0):
    """Get position of a cube by its TF frame name"""
    print(f"Getting {cube_name} position...")
    if not rospy.core.is_initialized():
        rospy.init_node('tf_xyz_fetcher', anonymous=True)
    
    listener = tf.TransformListener()
    try:
        print(f"Waiting for transform /world -> /{cube_name}...")
        listener.waitForTransform('/world', f'/{cube_name}', rospy.Time(0), rospy.Duration(timeout))
        
        trans, rot = listener.lookupTransform('/world', f'/{cube_name}', rospy.Time(0))
        print(f"{cube_name} position: {trans}")
        return trans, rot
    except Exception as e:
        print(f"Error getting transform for {cube_name}: {e}")
        return None

def execute_home_position(publisher, HOME_POSITION):
    # 2. RETURN TO HOME
    print("\n=== Returning to Home Position ===")
    publisher.publish_home_position(
        home_position=HOME_POSITION,
        execution_time=5.0
    )
    print("[Home] Waiting for home position...")
    rospy.sleep(7.0)  # Wait for home position completion
    print("[Home] Home position reached!")


def execute_pick(dmp_gen, bag_path, dmp_save_path, publisher, target_information, goal_information, execute_time_factor = 5, visualize = False):

    execute_home_position(publisher, HOME_POSITION)

    execute_dmp_pick(dmp_gen, bag_path, dmp_save_path, publisher, target_information, execute_time_factor = execute_time_factor, visualize = visualize, move_gripper = False)

    approach_and_grasp_cube(dmp_gen, publisher, target_information, approach_duration=3.0, grasp_duration=2.0, frequency=10.0, visualize=visualize)

    move_linearly_delta(dmp_gen=dmp_gen, publisher=publisher, target_position_delta=[-0.05, 0, 0.1], gripper_status="closed", duration=4.0, visualize=visualize)

    execute_home_position(publisher, HOME_POSITION)


def execute_place(dmp_gen, bag_path, dmp_save_path, publisher, target_information, goal_information, execute_time_factor = 5, visualize = False):

    execute_home_position(publisher, HOME_POSITION)

    execute_dmp_place(dmp_gen, bag_path, dmp_save_path, publisher, target_information, goal_information, execute_time_factor = execute_time_factor, visualize = visualize, move_gripper = False)

    approach_and_drop_cube(dmp_gen, publisher, target_information, goal_information, approach_duration=3.0, drop_duration=2.0, frequency=10.0, visualize=visualize)

    move_linearly_delta(dmp_gen=dmp_gen, publisher=publisher, target_position_delta=[-0.05, 0, 0.1], gripper_status="open", duration=4.0, visualize=visualize)

    execute_home_position(publisher, HOME_POSITION)


def execute_dmp_place(dmp_gen, bag_path, dmp_save_path, publisher, target_information, goal_information,
                    execute_time_factor=5, visualize=False, move_gripper = False):
    """Execute a motion (pick) with given parameters"""
    print(f"\n=== Executing pick motion ===")
    
    # Learn from bag
    print(f"Learning place motion from bag: {bag_path}")
    Y, transforms, joint_traj, gripper_traj = dmp_gen.learn_from_rosbag(
        bag_path, 
        '/gravity_compensation_controller/traj_joint_states'
    )
    
    # Save DMP
    dmp_gen.save_dmp(dmp_save_path)
    print(f"PLACE DMP saved to: {dmp_save_path}")
    
    # Get target position
    target_location = target_information["position"]
    target_name = target_information["name"]

    # Get goal position
    goal_location = goal_information["position"]
    goal_name = goal_information["name"]




    
    # Get start position
    current_joint_angles = get_current_joint_states()
    if current_joint_angles is None:
        print("ERROR: Could not get current robot joint angles. Aborting linear move.")
        return False
        
    current_transform = dmp_gen.chain.forward(current_joint_angles)
    start_position = current_transform[:3, 3]
    start_orientation = ptr.pqs_from_transforms(current_transform)[3:]
    print(f"Current EE Position: {start_position}")

    # Set start and goal
    new_start = dmp_gen.dmp.start_y.copy()
    new_goal = dmp_gen.dmp.goal_y.copy()

    new_start[0:3] = start_position
    new_start[3:] = start_orientation
    
    # Adjust goal position and orientation

    """
    if target_location[2] < 0.05:
        target_angle = -55
        x_offset_angle = -0.01
    elif target_location[2] < 0.9:
        target_angle = -45
        x_offset_angle = -0.015
    else:
        target_angle = -35
        x_offset_angle = -0.02
    """
    target_angle=-55
    x_offset_angle = -0.015
    

    # Calculate offset based on which cube is moved (with the dmp, we move over the cube and afterwards use a linear motion to move down)

    if target_name == "blue_cube":
        position_offset = [0.0, 0.0, 0.1]
    elif target_name == "red_cube":
        position_offset = [0.0, 0.0, 0.1]
    elif target_name == "green_cube":
        position_offset = [0.0, 0.0, 0.1]

    

    print(f"Original goal: {new_goal[:3]}")
    new_goal[:3] = np.array(goal_location) + np.array(position_offset) #+ np.array([x_offset_angle, 0.0, 0.0])
    print(f"New goal: {new_goal[:3]}")
    
    # Adjust goal orientation
    new_goal = ptr.transforms_from_pqs(new_goal)
    goal_orientation = create_orientation_in_xz_plane(target_angle)
    new_goal[:3,:3] = goal_orientation
    new_goal = ptr.pqs_from_transforms(new_goal)

    # Generate trajectory
    T, trajectory = dmp_gen.generate_trajectory(start_y=new_start, goal_y=new_goal)
    
    # Compute IK
    trajectory, IK_joint_trajectory, gripper_traj, T = dmp_gen.compute_IK_trajectory(
        trajectory, T, subsample_factor=2)
    
    # Apply smoothing
    window_size = 25
    if len(IK_joint_trajectory) > window_size:
        original_start = IK_joint_trajectory[0,:].copy()
        original_end = IK_joint_trajectory[-1,:].copy()

        smoothed_IK_joint_trajectory = np.zeros_like(IK_joint_trajectory)
        for i in range(IK_joint_trajectory.shape[1]):
            smoothed_IK_joint_trajectory[:, i] = np.convolve(IK_joint_trajectory[:, i], 
                                                           np.ones(window_size)/window_size, mode='same')

        smoothed_IK_joint_trajectory[0,:] = original_start
        smoothed_IK_joint_trajectory[-1,:] = original_end

        half_window = window_size // 2
        for i in range(IK_joint_trajectory.shape[1]):
            for j in range(half_window):
                alpha = j / float(half_window)
                smoothed_IK_joint_trajectory[j, i] = (1 - alpha) * original_start[i] + alpha * smoothed_IK_joint_trajectory[j, i]
            for j in range(half_window):
                alpha = j / float(half_window)
                idx_from_end = len(IK_joint_trajectory) - 1 - j
                smoothed_IK_joint_trajectory[idx_from_end, i] = (1 - alpha) * original_end[i] + alpha * smoothed_IK_joint_trajectory[idx_from_end, i]

        IK_joint_trajectory = smoothed_IK_joint_trajectory
        print(f"Applied moving average filter with window size {window_size} to IK trajectory.")
    else:
        print(f"Trajectory too short for smoothing (length {len(IK_joint_trajectory)})")

    # Visualize if requested
    
    if visualize:
        dmp_gen.visualize_trajectory(trajectory, IK_joint_trajectory)

    OPEN_GRIPPER_VAL = -0.007 
    CLOSED_GRIPPER_VAL = 0.005
    
    # Prepare full trajectory
    traj_length = min(IK_joint_trajectory.shape[0], len(gripper_traj) if gripper_traj is not None else IK_joint_trajectory.shape[0])
    IK_joint_trajectory = IK_joint_trajectory[:traj_length, :]
    
    if move_gripper is True:
        if gripper_traj is not None:
            gripper_traj = gripper_traj[:traj_length]
            full_trajectory = np.hstack((IK_joint_trajectory, -gripper_traj.reshape(-1, 1)))
        else:
            gripper_traj = np.zeros(traj_length)
            full_trajectory = np.hstack((IK_joint_trajectory, -gripper_traj.reshape(-1, 1)))
    else:
        if gripper_traj is not None:
            #gripper_traj = np.ones(traj_length)*gripper_traj[0]#/np.max(gripper_traj)*0.005
            #gripper_traj = gripper_traj[:traj_length]
            print(f"gripper_traj is not None")
            gripper_traj = np.ones(traj_length)*CLOSED_GRIPPER_VAL#np.linspace(gripper_traj[0], 0.005, traj_length)
            full_trajectory = np.hstack((IK_joint_trajectory, -gripper_traj.reshape(-1, 1)))
        else:
            gripper_traj = np.ones(traj_length)*CLOSED_GRIPPER_VAL
            full_trajectory = np.hstack((IK_joint_trajectory, -gripper_traj.reshape(-1, 1)))

    
    # Interpolate trajectory
    interpolated_traj, interpolated_time = interpolate_joint_trajectory(
        full_trajectory, T[:traj_length], target_freq=100.0)

    # Execute trajectory
    print(f"[pick] Starting trajectory execution...")
    
    # Clip trajectory to 95% of length to avoid oscillations in the learned motions
    clip_length = int(0.95 * len(interpolated_traj))
    arm_trajectory = interpolated_traj[:clip_length, :6]
    gripper_trajectory = interpolated_traj[:clip_length, 6]
    
    publisher.publish_trajectory(arm_trajectory, -gripper_trajectory, 
                               interpolated_time, execute_time_factor=execute_time_factor)
    
    # Wait for completion
    trajectory_execution_time = max(interpolated_time) * execute_time_factor
    print(f"[pick] Waiting {trajectory_execution_time:.2f} seconds for completion...")
    rospy.sleep(trajectory_execution_time + 2.0)
    
    print(f"[pick] Motion completed successfully!")
    return True



def execute_dmp_pick(dmp_gen, bag_path, dmp_save_path, publisher, target_information,
                    execute_time_factor=5, visualize=False, move_gripper = False):
    """Execute a motion (pick) with given parameters"""
    print(f"\n=== Executing pick motion ===")
    
    # Learn from bag
    print(f"Learning pick motion from bag: {bag_path}")
    Y, transforms, joint_traj, gripper_traj = dmp_gen.learn_from_rosbag(
        bag_path, 
        '/gravity_compensation_controller/traj_joint_states'
    )
    
    # Save DMP
    dmp_gen.save_dmp(dmp_save_path)
    print(f"PICK DMP saved to: {dmp_save_path}")
    
    # Get target position
    target_location = target_information["position"]
    target_name = target_information["name"]
    
    # Get start position
    current_joint_angles = get_current_joint_states()
    if current_joint_angles is None:
        print("ERROR: Could not get current robot joint angles. Aborting linear move.")
        return False
        
    current_transform = dmp_gen.chain.forward(current_joint_angles)
    start_position = current_transform[:3, 3]
    start_orientation = ptr.pqs_from_transforms(current_transform)[3:]
    print(f"Current EE Position: {start_position}")

    # Set start and goal
    new_start = dmp_gen.dmp.start_y.copy()
    new_goal = dmp_gen.dmp.goal_y.copy()

    new_start[0:3] = start_position
    new_start[3:] = start_orientation
    
    # Adjust goal position and orientation

    if target_location[2] < 0.05:
        target_angle = -55
        x_offset_angle = -0.01
    elif target_location[2] < 0.9:
        target_angle = -45
        x_offset_angle = -0.015
    else:
        target_angle = -35
        x_offset_angle = -0.02
    

    # Calculate offset based on which cube is moved (with the dmp, we move over the cube and afterwards use a linear motion to move down)

    if target_name == "blue_cube":
        position_offset = [0.0, 0.0, 0.05]
    elif target_name == "red_cube":
        position_offset = [0.0, 0.0, 0.055]
    elif target_name == "green_cube":
        position_offset = [0.0, 0.0, 0.065]

    

    print(f"Original goal: {new_goal[:3]}")
    new_goal[:3] = np.array(target_location) + np.array(position_offset) + np.array([x_offset_angle, 0.0, 0.0])
    print(f"New goal: {new_goal[:3]}")
    
    # Adjust goal orientation
    new_goal = ptr.transforms_from_pqs(new_goal)
    goal_orientation = create_orientation_in_xz_plane(target_angle)
    new_goal[:3,:3] = goal_orientation
    new_goal = ptr.pqs_from_transforms(new_goal)

    # Generate trajectory
    T, trajectory = dmp_gen.generate_trajectory(start_y=new_start, goal_y=new_goal)
    
    # Compute IK
    trajectory, IK_joint_trajectory, gripper_traj, T = dmp_gen.compute_IK_trajectory(
        trajectory, T, subsample_factor=2)
    
    # Apply smoothing
    window_size = 25
    if len(IK_joint_trajectory) > window_size:
        original_start = IK_joint_trajectory[0,:].copy()
        original_end = IK_joint_trajectory[-1,:].copy()

        smoothed_IK_joint_trajectory = np.zeros_like(IK_joint_trajectory)
        for i in range(IK_joint_trajectory.shape[1]):
            smoothed_IK_joint_trajectory[:, i] = np.convolve(IK_joint_trajectory[:, i], 
                                                           np.ones(window_size)/window_size, mode='same')

        smoothed_IK_joint_trajectory[0,:] = original_start
        smoothed_IK_joint_trajectory[-1,:] = original_end

        half_window = window_size // 2
        for i in range(IK_joint_trajectory.shape[1]):
            for j in range(half_window):
                alpha = j / float(half_window)
                smoothed_IK_joint_trajectory[j, i] = (1 - alpha) * original_start[i] + alpha * smoothed_IK_joint_trajectory[j, i]
            for j in range(half_window):
                alpha = j / float(half_window)
                idx_from_end = len(IK_joint_trajectory) - 1 - j
                smoothed_IK_joint_trajectory[idx_from_end, i] = (1 - alpha) * original_end[i] + alpha * smoothed_IK_joint_trajectory[idx_from_end, i]

        IK_joint_trajectory = smoothed_IK_joint_trajectory
        print(f"Applied moving average filter with window size {window_size} to IK trajectory.")
    else:
        print(f"Trajectory too short for smoothing (length {len(IK_joint_trajectory)})")

    # Visualize if requested
    
    if visualize:
        dmp_gen.visualize_trajectory(trajectory, IK_joint_trajectory)
    
    # Prepare full trajectory
    traj_length = min(IK_joint_trajectory.shape[0], len(gripper_traj) if gripper_traj is not None else IK_joint_trajectory.shape[0])
    IK_joint_trajectory = IK_joint_trajectory[:traj_length, :]
    
    if move_gripper is True:
        if gripper_traj is not None:
            gripper_traj = gripper_traj/np.max(gripper_traj)*-0.007
            gripper_traj = gripper_traj[:traj_length]
            full_trajectory = np.hstack((IK_joint_trajectory, -gripper_traj.reshape(-1, 1)))
        else:
            gripper_traj = np.zeros(traj_length)
            full_trajectory = np.hstack((IK_joint_trajectory, -gripper_traj.reshape(-1, 1)))
    else:
        if gripper_traj is not None:
            #gripper_traj = np.ones(traj_length)*gripper_traj[0]#/np.max(gripper_traj)*0.005
            #gripper_traj = gripper_traj[:traj_length]
            
            gripper_traj = np.linspace(gripper_traj[0], -0.007, traj_length)
            full_trajectory = np.hstack((IK_joint_trajectory, -gripper_traj.reshape(-1, 1)))
        else:
            gripper_traj = np.ones(traj_length)*-0.007
            full_trajectory = np.hstack((IK_joint_trajectory, -gripper_traj.reshape(-1, 1)))

    
    # Interpolate trajectory
    interpolated_traj, interpolated_time = interpolate_joint_trajectory(
        full_trajectory, T[:traj_length], target_freq=100.0)

    # Execute trajectory
    print(f"[pick] Starting trajectory execution...")
    
    # Clip trajectory to 95% of length to avoid oscillations in the learned motions
    clip_length = int(0.95 * len(interpolated_traj))
    arm_trajectory = interpolated_traj[:clip_length, :6]
    gripper_trajectory = interpolated_traj[:clip_length, 6]
    
    publisher.publish_trajectory(arm_trajectory, -gripper_trajectory, 
                               interpolated_time, execute_time_factor=execute_time_factor)
    
    # Wait for completion
    trajectory_execution_time = max(interpolated_time) * execute_time_factor
    print(f"[pick] Waiting {trajectory_execution_time:.2f} seconds for completion...")
    rospy.sleep(trajectory_execution_time + 2.0)
    
    print(f"[pick] Motion completed successfully!")
    return True




def get_current_joint_states(timeout=2.0):
    """Gets the current joint states of the robot arm."""
    try:
        print("Waiting for joint states...")
        joint_states_msg = rospy.wait_for_message(
            '/joint_states', 
            JointState, 
            timeout=timeout
        )
        # Assuming the first 6 joints are for the arm in the correct order
        print(f"Received joint states: {joint_states_msg.position[:6]}")
        return np.array(joint_states_msg.position[:6])
    except rospy.ROSException as e:
        rospy.logerr(f"Failed to get current joint states: {e}")
        return None



def approach_and_grasp_cube(dmp_gen, publisher, target_information, approach_duration=3.0,
                            grasp_duration=2.0, frequency=10.0, visualize=True):
    """
    Moves the gripper linearly to the cube's center, then closes the gripper.

    This function executes a two-phase trajectory:
    1. Arm moves from current location to the cube's center. Gripper remains open.
    2. Arm holds its position at the cube. Gripper closes.

    Args:
        dmp_gen (DMPMotionGenerator): The motion generator instance.
        publisher (GazeboTrajectoryPublisher): The trajectory publisher instance.
        cube_name (str): The name of the target cube (e.g., 'blue_cube').
        approach_duration (float): Time in seconds for the linear approach.
        grasp_duration (float): Time in seconds for the gripper to close.
        frequency (float): The frequency in Hz for the trajectory points.
    """

    # --- 1. Get Target and Current Positions ---
    target_location = target_information["position"]
    target_name = target_information["name"]

    print(f"\n=== Initiating Approach and Grasp for '{target_name}' ===")

    


    current_joint_angles = get_current_joint_states()
    if current_joint_angles is None:
        print("ERROR: Could not get current robot joint angles. Aborting grasp.")
        return False

    current_transform = dmp_gen.chain.forward(current_joint_angles)
    start_position = current_transform[:3, 3]
    start_orientation = current_transform[:3, :3]

    if target_name == "blue_cube":
        position_offset = [0.0, 0.0, -0.05]
    elif target_name == "red_cube":
        position_offset = [0.0, 0.0, -0.04]
    elif target_name == "green_cube":
        position_offset = [0.0, 0.0, -0.05]


    goal_position = np.array(start_position)+position_offset # The goal is the cube's center
    #goal_position[2] = goal_position[2]+offset_height
    print(f"Current EE Position: {start_position}")
    print(f"Target EE Position: {goal_position}")

    # --- 2. Phase 1: Generate Linear Approach Trajectory ---
    num_approach_steps = int(approach_duration * frequency)
    cartesian_approach_traj = np.zeros((num_approach_steps, 4, 4))

    for i in range(num_approach_steps):
        alpha = i / (num_approach_steps - 1.0)
        interp_position = (1 - alpha) * start_position + alpha * goal_position
        transform = np.eye(4)
        transform[:3, 3] = interp_position
        transform[:3, :3] = start_orientation
        cartesian_approach_traj[i] = transform

    print(f"Solving IK for the {num_approach_steps}-point approach path...")
    arm_approach_traj = dmp_gen.chain.inverse_trajectory(
        cartesian_approach_traj,
        initial_joint_angles=current_joint_angles,
        orientation_weight=1.0
    )

    if arm_approach_traj is None or len(arm_approach_traj) == 0:
        print("ERROR: Inverse kinematics failed for the approach path.")
        return False

    if visualize:
        dmp_gen.visualize_trajectory(cartesian_approach_traj, arm_approach_traj)

    OPEN_GRIPPER_VAL = -0.007 
    CLOSED_GRIPPER_VAL = 0.005

    # During approach, gripper is open. Positive values close the gripper.
    gripper_approach_traj = np.ones(num_approach_steps)*OPEN_GRIPPER_VAL

    # --- 3. Phase 2: Generate Grasping Trajectory (Arm is static) ---
    num_grasp_steps = int(grasp_duration * frequency)
    
    # Arm stays at the final position of the approach trajectory
    final_arm_pos = arm_approach_traj[-1]
    arm_grasp_traj = np.tile(final_arm_pos, (num_grasp_steps, 1))
    
    # Gripper closes from 0 (open) to 0.01 (closed).
    # These values might need tuning for your specific gripper controller.
    gripper_grasp_traj = np.linspace(OPEN_GRIPPER_VAL, CLOSED_GRIPPER_VAL, num_grasp_steps)

    # --- 4. Combine Trajectories ---
    full_arm_trajectory = np.vstack((arm_approach_traj, arm_grasp_traj))
    full_gripper_trajectory = np.concatenate((gripper_approach_traj, gripper_grasp_traj))

    total_duration = approach_duration + grasp_duration
    total_steps = num_approach_steps + num_grasp_steps
    timestamps = np.linspace(0, total_duration, total_steps)

    # --- 5. Publish and Wait ---
    print(f"Publishing combined {total_steps}-point trajectory to Gazebo...")
    publisher.publish_trajectory(
        full_arm_trajectory,
        full_gripper_trajectory,
        timestamps,
        execute_time_factor=1.0
    )

    print(f"Waiting {total_duration + 1.0:.2f} seconds for grasp to complete...")
    rospy.sleep(total_duration + 1.0)

    print(f"Approach and grasp for '{target_name}' completed.")
    return True


def approach_and_drop_cube(dmp_gen, publisher, target_information, goal_information, approach_duration=3.0, drop_duration=2.0, frequency=10.0, visualize=True):
    """
    Moves the gripper linearly to the cube's center, then opens the gripper.

    This function executes a two-phase trajectory:
    1. Arm moves from current location to the cube's center. Gripper remains closed.
    2. Arm holds its position at the cube. Gripper opens.

    Args:
        dmp_gen (DMPMotionGenerator): The motion generator instance.
        publisher (GazeboTrajectoryPublisher): The trajectory publisher instance.
        cube_name (str): The name of the target cube (e.g., 'blue_cube').
        approach_duration (float): Time in seconds for the linear approach.
        drop_duration (float): Time in seconds for the gripper to open.
        frequency (float): The frequency in Hz for the trajectory points.
    """
        # --- 1. Get Target and Current Positions ---
    target_location = target_information["position"]
    target_name = target_information["name"]

    goal_location = goal_information["position"]
    goal_name = goal_information["name"]

    print(f"\n=== Initiating Approach and Drop for '{target_name}' ===")

    


    current_joint_angles = get_current_joint_states()
    if current_joint_angles is None:
        print("ERROR: Could not get current robot joint angles. Aborting grasp.")
        return False

    current_transform = dmp_gen.chain.forward(current_joint_angles)
    start_position = current_transform[:3, 3]
    start_orientation = current_transform[:3, :3]

    if target_name == "blue_cube":
        position_offset = [0.0, 0.0, -0.05]
    elif target_name == "red_cube":
        position_offset = [0.0, 0.0, -0.05]
    elif target_name == "green_cube":
        position_offset = [0.0, 0.0, -0.05]

    
    if goal_name == "blue_cube":
        goal_offset = [0.0, 0.0, 0.01]
    elif goal_name == "red_cube":
        goal_offset = [0.0, 0.0, 0.015]
    elif goal_name == "green_cube":
        goal_offset = [0.0, 0.0, 0.02]
    elif goal_name == "ground":
        goal_offset = [0.0, 0.0, 0.01]
    


    


    goal_position = np.array(start_position)+position_offset +goal_offset# The goal is the cube's center
    #goal_position[2] = goal_position[2]+offset_height
    print(f"Current EE Position: {start_position}")
    print(f"Target EE Position: {goal_position}")

    # --- 2. Phase 1: Generate Linear Approach Trajectory ---
    num_approach_steps = int(approach_duration * frequency)
    cartesian_approach_traj = np.zeros((num_approach_steps, 4, 4))

    for i in range(num_approach_steps):
        alpha = i / (num_approach_steps - 1.0)
        interp_position = (1 - alpha) * start_position + alpha * goal_position
        transform = np.eye(4)
        transform[:3, 3] = interp_position
        transform[:3, :3] = start_orientation
        cartesian_approach_traj[i] = transform

    print(f"Solving IK for the {num_approach_steps}-point approach path...")
    arm_approach_traj = dmp_gen.chain.inverse_trajectory(
        cartesian_approach_traj,
        initial_joint_angles=current_joint_angles,
        orientation_weight=1.0
    )

    if arm_approach_traj is None or len(arm_approach_traj) == 0:
        print("ERROR: Inverse kinematics failed for the approach path.")
        return False

    if visualize:
        dmp_gen.visualize_trajectory(cartesian_approach_traj, arm_approach_traj)

    OPEN_GRIPPER_VAL = -0.007 
    CLOSED_GRIPPER_VAL = 0.005

    # During approach, gripper is open. Positive values close the gripper.
    gripper_approach_traj = np.ones(num_approach_steps)*CLOSED_GRIPPER_VAL

    # --- 3. Phase 2: Generate Grasping Trajectory (Arm is static) ---
    num_grasp_steps = int(drop_duration * frequency)
    
    # Arm stays at the final position of the approach trajectory
    final_arm_pos = arm_approach_traj[-1]
    arm_grasp_traj = np.tile(final_arm_pos, (num_grasp_steps, 1))
    
    # Gripper closes from 0 (open) to 0.01 (closed).
    # These values might need tuning for your specific gripper controller.
    
    gripper_grasp_traj = np.linspace(CLOSED_GRIPPER_VAL, OPEN_GRIPPER_VAL, num_grasp_steps)

    # --- 4. Combine Trajectories ---
    full_arm_trajectory = np.vstack((arm_approach_traj, arm_grasp_traj))
    full_gripper_trajectory = np.concatenate((gripper_approach_traj, gripper_grasp_traj))

    total_duration = approach_duration + drop_duration
    total_steps = num_approach_steps + num_grasp_steps
    timestamps = np.linspace(0, total_duration, total_steps)

    # --- 5. Publish and Wait ---
    print(f"Publishing combined {total_steps}-point trajectory to Gazebo...")
    publisher.publish_trajectory(
        full_arm_trajectory,
        full_gripper_trajectory,
        timestamps,
        execute_time_factor=1.0
    )

    print(f"Waiting {total_duration + 1.0:.2f} seconds for grasp to complete...")
    rospy.sleep(total_duration + 1.0)

    print(f"Approach and grasp for '{target_name}' completed.")
    return True


def move_linearly_delta(dmp_gen, publisher, target_position_delta, gripper_status = "open",
                             duration=4.0, frequency=10.0, visualize = False):
    """
    Moves the gripper in a linear path from its current position to a point
    directly above the specified cube.

    Args:
        dmp_gen (DMPMotionGenerator): The motion generator instance.
        publisher (GazeboTrajectoryPublisher): The trajectory publisher instance.
        cube_name (str): The name of the target cube (e.g., 'blue_cube').
        offset_height (float): The height in meters above the cube's center.
        duration (float): The desired time in seconds for the linear motion.
        frequency (float): The frequency in Hz for the trajectory points.
    """

    # 2. Get the current end-effector position
    current_joint_angles = get_current_joint_states()
    if current_joint_angles is None:
        print("ERROR: Could not get current robot joint angles. Aborting linear move.")
        return False
        
    current_transform = dmp_gen.chain.forward(current_joint_angles)
    start_position = current_transform[:3, 3]
    start_orientation = current_transform[:3, :3]
    print(f"Current EE Position: {start_position}") 

    # 3. Define the goal position
    # The goal is offset_height meters directly above the cube
    goal_position = start_position+np.array(target_position_delta) 
    print(f"Target EE Position: {goal_position}")

    # 4. Generate the linear Cartesian trajectory
    num_steps = int(duration * frequency)
    cartesian_trajectory = np.zeros((num_steps, 4, 4))
    
    for i in range(num_steps):
        # Linear interpolation for position
        alpha = i / (num_steps - 1.0)
        interp_position = (1 - alpha) * start_position + alpha * goal_position
        
        # Build the full 4x4 transform for this step
        transform = np.eye(4)
        transform[:3, 3] = interp_position
        transform[:3, :3] = start_orientation
        cartesian_trajectory[i] = transform

     # 5. Compute the Inverse Kinematics for the trajectory
    print(f"Solving IK for {num_steps} points in the linear path...")
    start_time = time.time()
    
    
    # Use the current joint angles as the initial guess for the IK solver
    joint_trajectory = dmp_gen.chain.inverse_trajectory(
        cartesian_trajectory, 
        initial_joint_angles=current_joint_angles,
        orientation_weight=1.0
    )
    
    print(f"IK solved in {time.time() - start_time:.2f} seconds.")

    if joint_trajectory is None or len(joint_trajectory) == 0:
        print("ERROR: Inverse kinematics failed for the linear path.")
        return False


    if visualize:
        dmp_gen.visualize_trajectory(cartesian_trajectory, joint_trajectory)
        
    # 6. Publish the trajectory to Gazebo
    timestamps = np.linspace(0, duration, num_steps)
    
    # For this motion, we assume the gripper state does not change.
    # We will create a placeholder of zeros for the gripper.
    if gripper_status == "open":
        gripper_trajectory = np.ones(num_steps)*-0.007
    elif gripper_status == "closed":
        gripper_trajectory = np.ones(num_steps)*0.005
    else:
        gripper_trajectory = np.zeros(num_steps)
    
    print("Publishing linear trajectory to Gazebo...")
    publisher.publish_trajectory(
        joint_trajectory, 
        gripper_trajectory, 
        timestamps, 
        execute_time_factor=1.0  # Execute in real-time
    )

    # 7. Wait for the motion to complete
    print(f"Waiting {duration + 1.0:.2f} seconds for linear motion to complete...")
    rospy.sleep(duration + 1.0) # Add a small buffer
    
    return True


def create_orientation_in_xz_plane(angle_z_with_world_x_deg: float) -> Rotation:
    """
    Creates an orientation where the local x and z axes are in the world x-z plane.

    The final orientation is uniquely determined by the angle between its local
    z-axis and the world x-axis. The resulting coordinate system is right-handed.

    Args:
        angle_z_with_world_x_deg: The desired angle in degrees between the local
                                  z-axis of the new orientation and the world's
                                  positive x-axis.

    Returns:
        A scipy.spatial.transform.Rotation object for the computed orientation.
    """
    angle_rad = np.radians(angle_z_with_world_x_deg)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    # 1. Define the local z-axis (z_new) in the world frame.
    # It lies in the world x-z plane at the specified angle to the world x-axis.
    z_new = np.array([cos_theta, 0.0, sin_theta])

    # 2. Define the local y-axis (y_new).
    # Since both the local x and z axes must be in the world x-z plane, the local
    # y-axis must be perpendicular to this plane. We choose the world's positive
    # y-axis to define a right-handed coordinate system.
    y_new = np.array([0.0, 1.0, 0.0])

    # 3. Define the local x-axis (x_new).
    # To complete the right-handed frame, x_new must be perpendicular to both
    # y_new and z_new. We find it using the cross product: x_new = y_new x z_new.
    x_new = np.cross(y_new, z_new)

    # 4. Construct the rotation matrix from the basis vectors (axes).
    # The columns of the rotation matrix are the new axes expressed in world coordinates.
    rotation_matrix = np.column_stack([x_new, y_new, z_new])

    return rotation_matrix


def get_state():

    """
    Categorizes cubes into positions A, B, or C based on their y-coordinate
    and sorts them based on their z-coordinate to determine stacking.

    Args:
        cube_positions: A dictionary with cube names as keys and their
                        (x, y, z) coordinates as values.

    Returns:
        A dictionary with keys "A", "B", "C" and values being lists of
        cube names, sorted by their z-coordinate (stacking order).
    """

    cube_names = ["blue_cube", "red_cube", "green_cube"]

    cube_positions = {}

    for cube_name in cube_names:
        cube_pos = None
        while cube_pos is None:

            cube_pos, _ = get_cube_position(cube_name)

        cube_positions[cube_name] = cube_pos

    # Temporary lists to hold tuples of (z_coordinate, cube_name)
    pos_a_temp = []
    pos_b_temp = []
    pos_c_temp = []

    # Categorize cubes based on the y-coordinate
    for name, (x, y, z) in cube_positions.items():
        if y < -0.04:
            pos_a_temp.append((z, name))
        elif -0.04 <= y <= 0.04:
            pos_b_temp.append((z, name))
        else: # y > 0.04
            pos_c_temp.append((z, name))

    # Sort each list based on the z-coordinate (the first element in the tuple)
    pos_a_temp.sort()
    pos_b_temp.sort()
    pos_c_temp.sort()

    # Extract the names to get the final sorted lists
    position_A = [name for z, name in pos_a_temp]
    position_B = [name for z, name in pos_b_temp]
    position_C = [name for z, name in pos_c_temp]

    print(f"position_A: {position_A}")
    print(f"position_B: {position_B}")
    print(f"position_C: {position_C}")

    return {"A": position_A, "B": position_B, "C": position_C}




def plan_motion(instruction):
    """
    Instructions can be e.g. MD1AB -> 1 = blue, move from A-> B
    "location" : A,B,C
    "position": x,y,z
    """

    if instruction[:2] != "MD" or instruction[3] not in "ABC" or instruction[4] not in "ABC":
        print("invalid instruction")
        return False

    cubenumber2name = {"1": "blue_cube", "2": "red_cube", "3":"green_cube"}

    cubenumber = instruction[2]
    cube_target = cubenumber2name[cubenumber]
    cube_target_pos, _ = get_cube_position(cube_target)
    if cube_target_pos is None:
        print(f"ERROR: Could not find cube '{cube_target}'. Aborting grasp.")
        return False

    target_information = {"name": cube_target, "position": cube_target_pos}


    goal_location = instruction[4]

    state = get_state()

    print(f"state: {state}")
    print(f"goal_location: {goal_location}")
    goal_location_state = state[goal_location]
    print(f"goal_location_state: {goal_location_state}")

    if not goal_location_state:
        if goal_location == "A":
            goal_position = [0.2, -0.12, 0.01]
        elif goal_location == "B":
            goal_position = [0.2, 0.0, 0.01]
        else:
            goal_position = [0.2, 0.12, 0.01]

        goal_information = {"name": "ground", "position": goal_position}

    else:
        cube_goal = goal_location_state[-1]
        print(f"cube_goal: {cube_goal}")
        cube_goal_position, _ = get_cube_position(cube_goal)

        goal_information = {"name": cube_goal, "position": cube_goal_position}

    

    return target_information, goal_information


if __name__ == "__main__":

    
    # Configuration
    urdf_path = '/root/catkin_ws/src/open_manipulator_friends/open_manipulator_6dof_description/urdf/open_manipulator_6dof.urdf'
    mesh_path = '/root/catkin_ws/src/open_manipulator_friends/open_manipulator_6dof_description/meshes'
    
    # Bag file paths
    pick_bag_path = '/root/catkin_ws/src/om_position_controller/recording/pick.bag'#'/root/catkin_ws/recordings_1905_new/pick_null_3.bag'#"/root/catkin_ws/recordings_14_05/pick_1.bag"#
    place_bag_path = '/root/catkin_ws/src/om_position_controller/recording/place.bag'  # Add your place bag path
    
    # DMP save paths
    pick_dmp_path = '/root/catkin_ws/src/om_position_controller/recording/pick_motion.pkl'
    place_dmp_path = '/root/catkin_ws/src/om_position_controller/recording/place_motion.pkl'
    




    
    # Home position
    #original:
    #home_position = [-0.03834952, -0.84062147, 1.26093221, 0.00613592, 1.97576725, -0.00460194]

    #Adjusted home position to avoid inverse kinematics issues
    HOME_POSITION = [-0.03834952, -0.44062147, 1.06093221, 0.00613592, 1.77576725, -0.00460194]
    
    print("=== Starting Pick and Place Operation ===")
    
    try:
        # Initialize components
        dmp_gen = DMPMotionGenerator(
            urdf_path, 
            mesh_path,
            base_link="world"
        )

#
        
        publisher = GazeboTrajectoryPublisher()
        rospy.sleep(2.0)

        with open("/root/catkin_ws/src/om_position_controller/TowerOfHanoi/final/next_move.txt") as f: 
            instruction = f.readlines()[0]
        
        target_information, goal_information = plan_motion(instruction)
        print(f"target_information: {target_information}")
        print(f"goal_information: {goal_information}")

        execute_pick(dmp_gen, pick_bag_path, pick_dmp_path, publisher, target_information, goal_information, execute_time_factor = 5, visualize = False)
        execute_place(dmp_gen, pick_bag_path, pick_dmp_path, publisher, target_information, goal_information, execute_time_factor = 5, visualize = False)

        
    except rospy.ROSInterruptException:
        print("[Main] ROS interrupted.")
    except Exception as e:
        print(f"[Main] Error during pick and place operation: {e}")
        import traceback
        traceback.print_exc()