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
from scipy.signal import savgol_filter
import rospy
from sensor_msgs.msg import JointState
import geometry_msgs
import sys
import moveit_commander
from scipy.spatial.transform import Rotation as R
#from pinocchio import pin
#print(f"{pin.__version__}")
#-------------------------------------- Classes --------------------------------------# 

class DMPMotionGenerator:
    def __init__(self, urdf_path, mesh_path=None, joint_names=None, base_link="world", end_effector_link="end_effector_link"):
        """
        Initialize DMP Motion Generator
        
        Parameters:
        -----------
        urdf_path : str
            Path to the URDF file
        mesh_path : str, optional
            Path to mesh files
        joint_names : list, optional
            List of joint names to use
        base_link : str
            Name of the base link
        end_effector_link : str
            Name of the end effector link
        """
        self.urdf_path = urdf_path
        self.mesh_path = mesh_path
        self.kin = self._load_kinematics(urdf_path, mesh_path)
        self.joint_names = joint_names or ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.base_link = base_link
        self.end_effector_link = end_effector_link
        self.chain = self.kin.create_chain(self.joint_names, base_link, end_effector_link)
        self.dmp = None
        self.IK_joint_trajectory = None
        
        self.gripper_trajectory = None
        
    def _load_kinematics(self, urdf_path, mesh_path=None):
        """Load robot kinematics from URDF"""
        with open(urdf_path, 'r') as f:
            return Kinematics(f.read(), mesh_path=mesh_path)

    def learn_from_rosbag(self, bag_path, joint_topic, dt=None, n_weights=10):
        """Learn DMP from rosbag recording"""
        transforms, joint_trajectory,gripper_trajectory, time_stamp = self._process_rosbag(bag_path, joint_topic)
        print(f"Start POSE-H-Matrix: {transforms[0]}")
        print(f"Start POSE-Joint: {joint_trajectory[0]}")

        # Convert transforms to PQS representation
        Y = ptr.pqs_from_transforms(transforms)
        if dt is None:
            dt = 1/self.frequency
        # Create and train DMP
        self.dmp = CartesianDMP(execution_time=max(time_stamp), dt=dt, n_weights_per_dim=n_weights)
        print(f"Start POSE: {Y[0]}")
        self.dmp.imitate(time_stamp, Y)
        self.gripper_trajectory = gripper_trajectory

        
        
        return Y, transforms, joint_trajectory, gripper_trajectory

    def _process_rosbag(self, bag_path, joint_topic):
        """Process rosbag and extract trajectories"""
        transforms = []
        joint_trajectory = []
        gripper_trajectory = []
        time_stamp = []
        
        print(f"Reading bag file: {bag_path}")
        bag = rosbag.Bag(bag_path)
        for topic, msg, t in bag.read_messages(topics=[joint_topic]):
            joint_pos = msg.position[:6]
            gripper_pos = msg.position[6]
            joint_trajectory.append(joint_pos)
            gripper_trajectory.append(gripper_pos)

            transforms.append(self.chain.forward(joint_pos))
            time_stamp.append(msg.header.stamp.to_sec())    
        bag.close()
        
        # Convert to numpy arrays
        
        transforms = np.array(transforms)
        joint_trajectory = np.array(joint_trajectory)
        gripper_trajectory = np.array(gripper_trajectory)
        time_stamp = np.array(time_stamp)
        
        dt = []
        for i in range(1, time_stamp.shape[0]):
            dt.append(time_stamp[i]- time_stamp[i-1])
        self.frequency = 1/ np.average(np.array(dt))
        # print(f"Average frequency: { self.frequency}")
        # First filter outliers
        positions = np.array([T[:3, 3] for T in transforms])
        mask, _ = self.remove_outliers_mad(positions, threshold=5.0)
        
        # Then normalize time (important to do it in this order)
        filtered_time = time_stamp[mask]
        normalized_time = filtered_time - filtered_time[0]
        
        # print(f"Shape of filtered transforms: {transforms[mask].shape}")
        # print(f"Shape of time stamp: {normalized_time.shape}")
        
        return transforms[mask], joint_trajectory[mask], gripper_trajectory[mask] , normalized_time

    def remove_outliers_mad(self, data, threshold=3.5):
        """Remove outliers using Median Absolute Deviation"""
        median = np.median(data, axis=0)
        diff = np.abs(data - median)
        mad = np.median(diff, axis=0)
        modified_z_score = 0.6745 * diff / (mad + 1e-6)
        mask = np.all(modified_z_score < threshold, axis=1)
        return mask, data[mask]



    def generate_trajectory(self, start_y=None, goal_y=None):
        """
        Generate trajectory using the learned DMP
        
        Parameters:
        -----------
        start_y : array-like, shape (7,)
            Start state in PQS format [x,y,z,qw,qx,qy,qz]
        goal_y : array-like, shape (7,)
            Goal state in PQS format [x,y,z,qw,qx,qy,qz]
        """
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
        return T, trajectory, Y

    def save_dmp(self, filepath):
        """Save the learned DMP to file"""
        if self.dmp is None:
            raise ValueError("No DMP model available to save")
        with open(filepath, 'wb') as f:
            pickle.dump((self.dmp, self.gripper_trajectory), f)
        print(f"DMP saved to {filepath}")

    def load_dmp(self, filepath):
        """Load a DMP from file"""
        print(f"Loading DMP from {filepath}")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.dmp = data[0]
            self.gripper_trajectory = data[1]
        print(f"DMP loaded successfully")
    
    def compute_IK_trajectory(self, trajectory, gripper_trajectory,  time_stamp, q0=None, subsample_factor=1):
        if q0 is None:
            q0 = np.array([0.0, -0.78, 1.5, 0., 0.8, 0.])
        
        # Subsample the trajectory if requested
        if subsample_factor > 1:
            subsampled_trajectory = trajectory[::subsample_factor]
            subsampled_time_stamp = time_stamp[::subsample_factor]
            subsampled_gripper_trajectory = gripper_trajectory[::subsample_factor]
            print(f"Subsampled time from {len(time_stamp)} to {len(subsampled_time_stamp)} points")
            print(f"Subsampled trajectory from {len(trajectory)} to {len(subsampled_trajectory)} points")
        else:
            subsampled_trajectory = trajectory
            subsampled_time_stamp = time_stamp
            subsampled_gripper_trajectory = gripper_trajectory
        print(f"Solving inverse kinematics for {len(subsampled_trajectory)} points...")
        
        start_time = time.time()
        
        # Use the same random state as in dmp_test_1.py
        random_state = np.random.RandomState(0)
        #joint_trajectory = self.chain.inverse_trajectory(
        #    subsampled_trajectory,  random_state=random_state, orientation_weight=0.5)

        joint_trajectory = self.chain.inverse_trajectory(
            subsampled_trajectory,  random_state=random_state, interval=np.pi/10, position_weight=1.0, orientation_weight=0.8, solver="COBYLA")

        print(f"IK solved in {time.time() - start_time:.2f} seconds")

        num_timesteps, num_joints = joint_trajectory.shape

    
    # Apply the filter to each joint trajectory (column) independently
        window_length=21
        polyorder = 3
        smoothed_joint_trajectory = np.zeros_like(joint_trajectory)
        for i in range(num_joints):
            # mode='interp' handles boundaries by fitting polynomials to points
            # available near the edges, which is often better than 'nearest' or 'mirror'
            # for trajectory smoothing. 'wrap' could be useful for cyclic joints.
            smoothed_joint_trajectory[:, i] = savgol_filter(
                joint_trajectory[:, i],
                window_length=window_length,
                polyorder=polyorder,
                mode='interp' # Or 'nearest', 'mirror', 'wrap' depending on boundary needs
            )
        print(f"num_joings: {num_joints}")
        joint_trajectory = smoothed_joint_trajectory
        
        
        return subsampled_trajectory, joint_trajectory, subsampled_gripper_trajectory ,subsampled_time_stamp


    def compute_IK_trajectory_moveit(self, trajectory, trajectory_quat, gripper_trajectory,  time_stamp, q0=None, subsample_factor=1):

        try:
            moveit_commander.roscpp_initialize(sys.argv)

            #rospy.init_node('cartesian_to_joint_trajectory_planner', anonymous=True)

            robot = moveit_commander.RobotCommander()

            #scene = moveit_commander.PlanningSceneInterface()

            group_name = "arm"
            move_group = moveit_commander.MoveGroupCommander(group_name)
            print(f"move_group")
            eef_link = move_group.get_end_effector_link()
            print(f"eef_link")
            
            # Subsample the trajectory if requested
            if subsample_factor > 1:
                subsampled_trajectory = trajectory[::subsample_factor]
                subsampled_trajectory_quat = trajectory_quat[::subsample_factor]
                subsampled_time_stamp = time_stamp[::subsample_factor]
                subsampled_gripper_trajectory = gripper_trajectory[::subsample_factor]
                print(f"Subsampled time from {len(time_stamp)} to {len(subsampled_time_stamp)} points")
                print(f"Subsampled trajectory from {len(trajectory)} to {len(subsampled_trajectory)} points")
            else:
                subsampled_trajectory = trajectory
                subsampled_time_stamp = time_stamp
                subsampled_gripper_trajectory = gripper_trajectory
            print(f"Solving inverse kinematics for {len(subsampled_trajectory)} points...")

            print(f"start numpy_array_to_pose_list")
            waypoints = numpy_array_to_pose_list(subsampled_trajectory_quat)
            print(f"end numpy_array_to_pose_list")

            print(f"waypoints[0]{waypoints[0]}")

            eef_step = 0.001
            jump_threshold = 0.0
            avoid_collisions = True

            (plan, fraction) = move_group.compute_cartesian_path(
                                        waypoints = waypoints,   # waypoints to follow
                                        eef_step = eef_step,    # eef_step
                                        #jump_threshold = jump_threshold, # jump_threshold
                                        avoid_collisions = avoid_collisions, # avoid_collisions
                                        path_constraints = None
                                        )
            print(f"Solved inverse kinematics for {len(subsampled_trajectory)} points...")
            joint_trajectory = plan.joint_trajectory
            print(f"len(joint_trajectory.points){len(joint_trajectory.points)}")
            # Extract the 'positions' list from each JointTrajectoryPoint
            # point.positions is already an ordered list/tuple of joint values for that timestep
            list_of_positions = [point.positions for point in joint_trajectory.points]

            # Convert the list of lists/tuples directly into a NumPy array
            # NumPy automatically determines the shape (num_timesteps, num_joints)
            joint_trajectory = np.array(list_of_positions)
            print(f"joint_trajectory.shape{joint_trajectory.shape}")

        
            #num_timesteps, num_joints = joint_trajectory.shape

            """
            # Apply the filter to each joint trajectory (column) independently
                window_length=21
                polyorder = 3
                smoothed_joint_trajectory = np.zeros_like(joint_trajectory)
                for i in range(num_joints):
                    # mode='interp' handles boundaries by fitting polynomials to points
                    # available near the edges, which is often better than 'nearest' or 'mirror'
                    # for trajectory smoothing. 'wrap' could be useful for cyclic joints.
                    smoothed_joint_trajectory[:, i] = savgol_filter(
                        joint_trajectory[:, i],
                        window_length=window_length,
                        polyorder=polyorder,
                        mode='interp' # Or 'nearest', 'mirror', 'wrap' depending on boundary needs
                    )
                print(f"num_joings: {num_joints}")
                joint_trajectory = smoothed_joint_trajectory
            """
        except rospy.ROSInterruptException:
            pass
        except Exception as e:
            rospy.logerr("An error occurred: %s", e)
            moveit_commander.roscpp_shutdown()

        
        return subsampled_trajectory, subsampled_trajectory_quat, joint_trajectory, subsampled_gripper_trajectory ,subsampled_time_stamp


    def compute_IK_trajectory_pinocchio(self, trajectory, time_stamp, gripper_trajectory, subsample_factor=1, q0 = None):

        model = pin.buildModelFromUrdf(self.urdf_path)

        data = model.createData()

        end_effector_frame_id = model.getFrameId(self.end_effector_link)

        if q0 is None:
            q0 =  np.array([-0.09203885, 0.09817477, 1.59227204, -0.0076699,   1.40972841,  0.0322136, 0, 0])        

        if subsample_factor > 1:
            subsampled_trajectory = trajectory[::subsample_factor]
            subsampled_trajectory_quat = trajectory_quat[::subsample_factor]
            subsampled_time_stamp = time_stamp[::subsample_factor]
            subsampled_gripper_trajectory = gripper_trajectory[::subsample_factor]
            print(f"Subsampled time from {len(time_stamp)} to {len(subsampled_time_stamp)} points")
            print(f"Subsampled trajectory from {len(trajectory)} to {len(subsampled_trajectory)} points")
        else:
            subsampled_trajectory = trajectory
            subsampled_time_stamp = time_stamp
            subsampled_gripper_trajectory = gripper_trajectory
        print(f"Solving inverse kinematics for {len(subsampled_trajectory)} points...")

        joint_configurations = []

        # Initial guess for the joint configuration (can be the robot's neutral pose or a previous solution)
        q_current = q0#pin.neutral(model)

        # Iterate through the time points and solve IK
        for i in range(len(subsampled_time_stamp)):
            target_pose = pin.SE3(subsampled_trajectory[i])
            

            # Solve the IK problem
            q_result = self._solve_IK_pinocchio_manipulability(target_pose, model, data, q_current, end_effector_frame_id)

            if q_result is not None:
                joint_configurations.append(q_result)
                q_current = q_result  # Use the previous solution as the next initial guess
            else:
                print(f"Warning: Ik failed at time {subsampled_time_stamp[i]}")
                joint_configurations.append(q_current)

        joint_trajectory = np.array(joint_configurations)


        return subsampled_trajectory, joint_trajectory, subsampled_gripper_trajectory ,subsampled_time_stamp


    def _solve_IK_pinocchio(self, desired_pose_se3, model, data, initial_q, end_effector_frame_id):
        """
        Solves Inverse Kinematics for the given desired_pose.

        :param desired_pose_se3: The desired pose as a pinocchio.SE3 object
        :param initial_q: Initial guess for joint configuration. If None, uses model.neutralConfiguration
        :return: The solved joint configuration q, or None if not converged.
        """
        q = initial_q if initial_q is not None else pin.neutral(self.model)

        # IK parameters
        EPS = 1e-5  # Tolerance for error norm
        IT_MAX = 1000  # Maximum iterations
        DT = 1e-1  # Integration step (tune this)
        DAMP = 1e-2# Damping factor for pseudo-inverse (tune this)

        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        for i in range(IT_MAX):
            # Compute current pose of the end-effector
            current_pose = data.oMf[end_effector_frame_id]#end_effector_frame_i

            # Compute error vector (in the local frame of the current_pose)
            # This error is the twist that would bring current_pose to desired_pose_se3
            error = pin.log6(current_pose.inverse() * desired_pose_se3).vector
            # print(f"Iteration {i}: error norm = {np.linalg.norm(error)}")

            if np.linalg.norm(error) < EPS:
                print(f"Convergence achieved after {i} iterations.")
                return q

            # Compute Jacobian in the local frame of the end-effector
            # (consistent with the error computed above)
            J = pin.computeFrameJacobian(model, data, q, end_effector_frame_id, pin.LOCAL)#

            # Solve for joint velocities using damped pseudo-inverse
            #J vq = error  => vq = J_pinv * error
            lambda_ = DAMP * np.eye(6) # Damping matrix
            vq = J.T @ np.linalg.solve(J @ J.T + lambda_, error)
            # Simpler pseudo-inverse (can be less stable near singularities)
            #vq = np.linalg.pinv(J) @ error

            # Update joint configuration
            q = pin.integrate(model, q, vq * DT)
            
            # Recompute forward kinematics and frame placements for the next iteration
            pin.forwardKinematics(model, data, q) # Essential to update data.oMf for next Jacobian & error
            pin.updateFramePlacements(model, data)


        print(f"IK failed to converge after {IT_MAX} iterations. Final error norm: {np.linalg.norm(error)}")
        return None


    def _solve_IK_pinocchio_manipulability(self, desired_pose_se3, model, data, initial_q, end_effector_frame_id, k_manip=0.01):
        """
        Solves Inverse Kinematics for the given desired_pose.

        :param desired_pose_se3: The desired pose as a pinocchio.SE3 object
        :param initial_q: Initial guess for joint configuration. If None, uses model.neutralConfiguration
        :return: The solved joint configuration q, or None if not converged.
        """
        q = initial_q if initial_q is not None else pin.neutral(self.model)

        # IK parameters
        EPS = 1e-5  # Tolerance for error norm
        IT_MAX = 10000  # Maximum iterations
        DT = 1e-1  # Integration step (tune this)
        DAMP = 1e-2# Damping factor for pseudo-inverse (tune this)

        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        for i in range(IT_MAX):
            # Compute current pose of the end-effector
            current_pose = data.oMf[end_effector_frame_id]#end_effector_frame_i

            # Compute error vector (in the local frame of the current_pose)
            # This error is the twist that would bring current_pose to desired_pose_se3
            error = pin.log6(current_pose.inverse() * desired_pose_se3).vector
            # print(f"Iteration {i}: error norm = {np.linalg.norm(error)}")

            if np.linalg.norm(error) < EPS:
                print(f"Convergence achieved after {i} iterations.")
                return q

            # Compute Jacobian in the local frame of the end-effector
            # (consistent with the error computed above)
            J = pin.computeFrameJacobian(model, data, q, end_effector_frame_id, pin.LOCAL)#

            try:
                J_pinv = np.linalg.pinv(J) # Damped pseudo-inverse could be more robust:
                                         # J_pinv = J.T @ np.linalg.solve(J @ J.T + DAMP * np.eye(J.shape[0]), np.eye(J.shape[0]))
            except np.linalg.LinAlgError:
                print("Singular Jacobian encountered for primary task.")
                return None # Or handle differently
            
            vq_primary = J_pinv @ error

            # Secondary task: Manipulability (using heuristic)
            if model.nv > J.shape[0]: # Only if redundant
                # Calculate gradient for manipulability (heuristic: joint centering)
                # For Yoshikawa's actual gradient: this would involve d(sqrt(det(JJT)))/dq, which needs dJ/dq.
                # This is complex. Pinocchio could potentially do this with its autodiff bindings (CasADi/CppAD)
                # or by manually deriving and implementing it using Pinocchio's derivative functions.
                # For now, let's use a simpler joint centering heuristic:
                grad_manip_heuristic = self.compute_manipulability_gradient_heuristic(q, model)

                # Null space projector
                N = np.eye(model.nv) - J_pinv @ J
                
                # Null space velocity for manipulability
                vq_secondary = N @ (k_manip * grad_manip_heuristic)
                
                # Total joint velocity
                vq = vq_primary + vq_secondary
            else:
                # Not redundant, no null space for secondary task
                vq = vq_primary
            
            # Debugging: print current manipulability
            # current_manip = self.compute_yoshikawa_manipulability(J)
            # print(f"Iter {i}: Error Norm: {np.linalg.norm(error):.4f}, Manipulability: {current_manip:.4f}")


            q = pin.integrate(model, q, vq * DT)
            
            pin.forwardKinematics(model, data, q) # Update for next iteration
            pin.updateFramePlacements(model, data)


        print(f"IK failed to converge after {IT_MAX} iterations. Final error norm: {np.linalg.norm(error)}")
        return None


    def compute_manipulability_gradient_heuristic(self, q, model):
        """
        Heuristic to push joints away from their limits, which often
        improves manipulability. A true gradient of Yoshikawa's index
        is more complex.
        Objective: minimize sum of ((q_i - q_mid_i) / (q_range_i))^2
        Gradient: d_obj/d_q_i = 2 * (q_i - q_mid_i) / (q_range_i)^2
        We want to move in the direction that *increases* manipulability,
        so for joint centering, we might want to move towards the center.
        If q_mid is the 'most manipulable' point for a joint, the gradient
        to move towards it is -(q - q_mid).
        """
        # A simple joint centering gradient
        # This encourages joints to stay near the middle of their range.
        q_mid = (model.lowerPositionLimit + model.upperPositionLimit) / 2.0
        # Normalize by range to make gains more consistent, or use a simpler form
        # For joints with no limits, this component would be zero or needs handling
        # if limits are +/- infinity.
        grad = -(q - q_mid)

        # Handle potential +/- inf in limits if not all joints are bounded
        for i in range(model.nv):
            if np.isinf(model.lowerPositionLimit[i]) or \
               np.isinf(model.upperPositionLimit[i]):
                grad[i] = 0.0 # No centering for unbounded joints
            # Could add a small value if q is exactly at q_mid to avoid zero grad if needed
        return grad

    def compute_yoshikawa_manipulability(self, J):
        """Computes Yoshikawa's manipulability index: sqrt(det(J J^T))"""
        if J.shape[0] < J.shape[1]: # Redundant manipulator, J is wide
            # For non-square J (typically rows < cols for redundancy), J J^T is square.
             pass # This is the standard case for redundant manipulators
        elif J.shape[0] > J.shape[1]: # Overconstrained, J is tall
            # Consider J^T J instead if that makes sense for the problem
            # or acknowledge that the standard Yoshikawa index might not be well-defined
            # or always positive.
            # For IK, usually J.shape[0] (task dimension) <= J.shape[1] (robot DoF)
            return 0.0 # Or handle appropriately
        if J.shape[0] == 0: # No task Jacobian
            return 0.0

        try:
            # Ensure J J^T is well-conditioned enough for determinant
            JJT = J @ J.T
            # Add small identity matrix for numerical stability if det is close to zero
            # JJT += np.eye(JJT.shape[0]) * 1e-9
            det_JJT = np.linalg.det(JJT)
            if det_JJT < 0: # Should ideally not happen for J J^T if J is real
                # This can occur due to numerical issues if JJT is very close to singular
                return 0.0
            return np.sqrt(det_JJT)
        except np.linalg.LinAlgError:
            return 0.0 # Singular or other issue





   
    def _smooth_trajectory(self, trajectory, window_size=5):
        """Apply moving average smoothing to trajectory"""
        smoothed = np.copy(trajectory)
        half_window = window_size // 2
        
        for i in range(len(trajectory)):
            # Calculate window indices with boundary handling
            start = max(0, i - half_window)
            end = min(len(trajectory), i + half_window + 1)
            
            # Calculate average for each component of the pose
            for row in range(4):
                for col in range(4):
                    if row < 3 and col < 3:  # Only smooth rotation part
                        smoothed[i, row, col] = np.mean(trajectory[start:end, row, col])
                    elif col == 3:  # Position part
                        smoothed[i, row, col] = np.mean(trajectory[start:end, row, col])
        
        return smoothed

    def compute_IK_trajectory_KDL(self, trajectory, time_stamp, q0=None, max_iterations=1000, eps=1e-2):
        # Import necessary KDL modules
        try:
            import PyKDL
            from urdf_parser_py.urdf import URDF
            from kdl_parser_py.urdf import treeFromUrdfModel
        except ImportError:
            print("Error: PyKDL or URDF parser modules not found. Install with:")
            print("sudo apt-get install python3-pyKDL ros-noetic-kdl-parser-py ros-noetic-urdfdom-py")
            raise

        if q0 is None:
            q0 = np.array([0.0, -0.78, 1.5, 0., 0.8, 0.])
        
        start_time = time.time()
        
        # Load robot model from URDF
        robot_model = URDF.from_xml_file(self.urdf_path)
        success, kdl_tree = treeFromUrdfModel(robot_model)
        if not success:
            raise ValueError("Failed to construct KDL tree from URDF")
        
        # Create KDL Chain
        kdl_chain = kdl_tree.getChain(self.base_link, self.end_effector_link)
        num_joints = kdl_chain.getNrOfJoints()
        print(f"num_joints{num_joints}")
        
        # Create KDL IK solvers
        fk_solver = PyKDL.ChainFkSolverPos_recursive(kdl_chain)
        ik_vel_solver = PyKDL.ChainIkSolverVel_pinv(kdl_chain)

        # Create joint limit arrays - initially set all to max range for simplicity
        # In a real application, you should get these from the URDF
        lower_limits = PyKDL.JntArray(num_joints)
        upper_limits = PyKDL.JntArray(num_joints)
        # Get joint limits from URDF
        for i, joint in enumerate(self.joint_names):
            # Find the joint in the robot model
            urdf_joint = None
            for j in robot_model.joints:
                if j.name == joint:
                    urdf_joint = j
                    break
            
            if urdf_joint and urdf_joint.limit:
                lower_limits[i] = urdf_joint.limit.lower
                upper_limits[i] = urdf_joint.limit.upper
            else:
                # Default limits if not found
                lower_limits[i] = -3.14
                upper_limits[i] = 3.14
        
        # Create the IK position solver with joint limits
        ik_solver = PyKDL.ChainIkSolverPos_NR_JL(
            kdl_chain, lower_limits, upper_limits, fk_solver, ik_vel_solver, 
            max_iterations, eps
        )
        
        # Initialize joint trajectory array
        joint_trajectory = np.zeros((len(trajectory), num_joints))
        print(f"joint_trajectory.shape{joint_trajectory.shape}")
        
        # Set initial joint positions
        q_kdl = PyKDL.JntArray(num_joints)
        for i in range(min(len(q0), num_joints)):
            q_kdl[i] = q0[i]
        
        # Smooth the trajectory
        # smooth_traj = self._smooth_trajectory(trajectory)
        
        
        # Solve IK for each point in the trajectory
        for i in range(len(trajectory)):
            # Extract current pose
            pose = trajectory[i]
            
            # Convert to KDL Frame
            frame = PyKDL.Frame(
                PyKDL.Rotation(
                    pose[0, 0], pose[0, 1], pose[0, 2],
                    pose[1, 0], pose[1, 1], pose[1, 2],
                    pose[2, 0], pose[2, 1], pose[2, 2]
                ),
                PyKDL.Vector(pose[0, 3], pose[1, 3], pose[2, 3])
            )
            
            # Prepare output joint array
            q_out = PyKDL.JntArray(num_joints)
            
            # Solve IK
            result = ik_solver.CartToJnt(q_kdl, frame, q_out)
            
            if result < 0:
                print(f"Warning: IK failed at point {i} with error code {result}")
                # If the first point fails, use initial guess
                if i == 0:
                    for j in range(num_joints):
                        q_out[j] = q_kdl[j]
                # Otherwise use previous solution
                else:
                    for j in range(num_joints):
                        q_out[j] = joint_trajectory[i-1, j]
            
            # Store the solution
            for j in range(num_joints):
                joint_trajectory[i, j] = q_out[j]
            
            # Use this solution as the seed for the next point
            q_kdl = q_out
            
            # Progress indicator for long trajectories
            if i % 50 == 0 and i > 0:
                print(f"Solved {i}/{len(trajectory)} points...")
        
        print(f"KDL IK solved in {time.time() - start_time:.2f} seconds")
        
        return trajectory, joint_trajectory, time_stamp
        
    
    def visualize_trajectory(self, trajectory, joint_trajectory, q0=None ):
        """
        Visualize the generated trajectory with optional subsampling
        
        Parameters:
        -----------
        trajectory : array-like
            The trajectory to visualize as homogeneous transformation matrices
        q0 : array-like, optional
            Initial joint configuration for inverse kinematics
        subsample_factor : int, optional
            Factor by which to subsample the trajectory. 
            1 means use all points, 2 means use every second point, etc.
        """
        
        print(f"Plotting trajectory...")
        fig = pv.figure()
        fig.plot_transform(s=0.3)
        
        # Use the same whitelist as in dmp_test_1.py
        graph = fig.plot_graph(
            self.kin.tm, "world", show_visuals=False, show_collision_objects=True,
            show_frames=True, s=0.1, whitelist=[self.base_link, self.end_effector_link])

        # Plot start and end pose for clarity
        fig.plot_transform(trajectory[0], s=0.15)
        fig.plot_transform(trajectory[-1], s=0.15)
        
        # Always show the full trajectory in the visualization
        pv.Trajectory(trajectory, s=0.05).add_artist(fig)
        
        fig.view_init()
        fig.animate(
            animation_callback, len(trajectory), loop=True,
            fargs=(graph, self.chain, joint_trajectory))
        fig.show()


class ROSTrajectoryPublisher:
    def __init__(self, joint_names, topic_name='/gravity_compensation_controller/traj_joint_states', rate_hz=20):
        #rospy.init_node("dmp_trajectory_publisher", anonymous=True)
        self.publisher = rospy.Publisher(topic_name, JointState, queue_size=10)
        
        joint_names.append("gripper")
        # print(f"joint names: {joint_names}")
        self.joint_names = joint_names
        self.rate = rospy.Rate(rate_hz)
        print(f"[ROS] Initialized publisher on topic {topic_name} at {rate_hz}Hz")

    def publish_trajectory(self, joint_trajectory, timestamps):
        """
        Publishes joint trajectory as JointState messages at fixed rate.

        Parameters:
        -----------
        joint_trajectory : np.ndarray
            Interpolated joint trajectory (M, D)
        timestamps : np.ndarray
            Corresponding timestamps (M,)
        """
        start_time = rospy.Time.now()
        for i in range(len(joint_trajectory)):
            if rospy.is_shutdown():
                break
            msg = JointState()
            msg.header.stamp = start_time + rospy.Duration.from_sec(timestamps[i] - timestamps[0])
            msg.name = self.joint_names
            position = joint_trajectory[i].tolist()
            
            # position.append(0.0) # gripper
            # print(f"Position: {position}")
            vel_eff = np.zeros(7).tolist()
            msg.velocity =  vel_eff   
            msg.effort = vel_eff
            # print(f"velocity: {vel_eff}")
            msg.position = position
            self.publisher.publish(msg)
            self.rate.sleep()

# -------------------------------------- Helper functions --------------------------------------# 
def animation_callback(step, graph, chain, joint_trajectory):
    """Animation callback for visualization"""
    chain.forward(joint_trajectory[step])
    graph.set_data()
    return graph

def save_trajectory_data(joint_trajectory, timestamps, filepath):
    """
    Save trajectory data to a pickle file

    Parameters:
    -----------
    joint_trajectory : np.ndarray
        Joint trajectory array (N, D)
    timestamps : np.ndarray
        Timestamps array (N,)
    filepath : str
        Path to save the pickle file
    """
    data = {
        'trajectory': joint_trajectory,
        'timestamps': timestamps
    }
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"[SAVE] Trajectory data saved to {filepath}")

def load_trajectory_data(filepath):
    """
    Load trajectory data from a pickle file

    Parameters:
    -----------
    filepath : str
        Path to load the pickle file

    Returns:
    --------
    joint_trajectory : np.ndarray
        Loaded joint trajectory
    timestamps : np.ndarray
        Loaded timestamps
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    joint_trajectory = data['trajectory']
    timestamps = data['timestamps']
    print(f"[LOAD] Loaded trajectory from {filepath} (length={len(joint_trajectory)})")
    return joint_trajectory, timestamps

def interpolate_joint_trajectory(joint_traj,  time_stamps, target_freq=20.0):
    """
    Interpolate joint trajectory to the target frequency

    Parameters:
    -----------
    joint_traj : np.ndarray
        Original joint positions (N, D)
    time_stamps : np.ndarray
        Original timestamps (N,)
    target_freq : float
        Target frequency in Hz

    Returns:
    --------
    interp_traj : np.ndarray
        Interpolated joint trajectory (M, D)
    new_timestamps : np.ndarray
        New timestamps (M,)
    """
    num_joints = joint_traj.shape[1]
    duration = time_stamps[-1] - time_stamps[0]
    num_samples = int(duration * target_freq)
    new_timestamps = np.linspace(time_stamps[0], time_stamps[-1], num_samples)
    
    interp_traj = np.zeros((num_samples, num_joints))
    for i in range(num_joints):
        interpolator = interp1d(time_stamps, joint_traj[:, i], kind='linear', fill_value="extrapolate")
        #interpolator = interp1d(time_stamps, joint_traj[:, i], kind='cubic', fill_value="extrapolate")
        interp_traj[:, i] = interpolator(new_timestamps)
    
    return interp_traj, new_timestamps

def numpy_array_to_pose_list(pose_array_nx7):
        """
        Converts an Nx7 NumPy array into a list of geometry_msgs.msg.Pose objects.
        Assumes each row is [x, y, z, qx, qy, qz, qw].

        Args:
            pose_array_nx7 (np.ndarray): An Nx7 NumPy array where N is the number of poses.

        Returns:
            list[geometry_msgs.msg.Pose]: A list of Pose objects.

        Raises:
            ValueError: If the input array does not have 7 columns.
        """
        waypoints = []
    
        num_rows, num_cols = pose_array_nx7.shape
    


        if num_cols != 7:
            raise ValueError(f"Input NumPy array must have 7 columns (x, y, z, qx, qy, qz, qw), but got {num_cols}")

        for i in range(num_rows):
            pose_msg = geometry_msgs.msg.Pose()

            # --- Position ---
            pose_msg.position.x = pose_array_nx7[i, 0]
            pose_msg.position.y = pose_array_nx7[i, 1]
            pose_msg.position.z = pose_array_nx7[i, 2]

            # --- Orientation (Quaternion) ---
            pose_msg.orientation.x = pose_array_nx7[i, 4]
            pose_msg.orientation.y = pose_array_nx7[i, 5]
            pose_msg.orientation.z = pose_array_nx7[i, 6]
            pose_msg.orientation.w = pose_array_nx7[i, 3]

            waypoints.append(pose_msg)

        return waypoints








# -------------------------------------- MAIN --------------------------------------# 
if __name__ == "__main__":
    # Example usage:
    urdf_path = '/root/catkin_ws/src/open_manipulator_friends/open_manipulator_6dof_description/urdf/open_manipulator_6dof.urdf'
    mesh_path = '/root/catkin_ws/src/open_manipulator_friends/open_manipulator_6dof_description/meshes'
    #bag_path = '/root/catkin_ws/recordings_30_04/pick_front.bag'
    bag_path=sys.argv[1]

    # Initialize the motion generator with the same base_link as dmp_test_1.py
    dmp_gen = DMPMotionGenerator(
        urdf_path, 
        mesh_path,
        base_link="world"
    )

    # Learn from demonstration
    Y, transforms, joint_traj, gripper_traj = dmp_gen.learn_from_rosbag(
        bag_path, 
        '/gravity_compensation_controller/traj_joint_states'
    )
    # Save the learned DMP if needed
    dmp_gen.save_dmp('/root/catkin_ws/dmp/learned_pick_motion.pkl')
    #dmp_gen.load_dmp('/root/catkin_ws/recordings/learned_pick_motion.pkl')
    
    ## Generate new trajectory
    
    # Define new goal
    new_start = dmp_gen.dmp.start_y.copy()
    new_goal = dmp_gen.dmp.goal_y.copy()

    orig_goal = dmp_gen.dmp.goal_y.copy()
    
    new_start[:3] += np.array([0.00, 0.00,0.0])
    #new_start[:3] = np.array([0.05, 0.05,0.01])
    
    new_goal[:3] += np.array([0.0, 0.0, 0.0])  # Modify position
    #new_goal[:3] = np.array([0.125, 0.125, 0.05])

    #new_goal[3:] = np.array([0, 0, 1, 0])


    #new_goal[3:] = calculate_desired_orientation(new_goal[:3])

    print(f"New start: {new_start}")
    print(f"New goal: {new_goal}")
    # Generate
    T, trajectory, trajectory_quat = dmp_gen.generate_trajectory(start_y=new_start, goal_y=new_goal)

    
    #data = np.hstack((trajectory_quat, T))
    np.savetxt("cartesian_trajectory_data_fail_trajectory.txt", trajectory_quat)
    np.savetxt("cartesian_trajectory_data_fail_time.txt", T)


    q0 =  np.array([-0.09203885, 0.09817477, 1.59227204, -0.0076699,   1.40972841,  0.0322136 ])#dmp_gen.chain.inverse(new_start, initial_joint_angles = np.array([0.0, -0.78, 1.5, 0., 0.8, 0.]))
    # Visualize the trajectory
    
    trajectory, IK_joint_trajectory, gripper_traj ,T = dmp_gen.compute_IK_trajectory(trajectory, gripper_traj, T ,q0 =q0, subsample_factor=5)
    #trajectory, IK_joint_trajectory, gripper_traj ,T = dmp_gen.compute_IK_trajectory_pinocchio(trajectory, gripper_traj, T, subsample_factor=2)


    transform_from_IK = []
    print(f"trajectory[-1]{trajectory[-1]}")
    print(f"IK_joint_trajectory.shape {IK_joint_trajectory.shape}")
    for i in range(IK_joint_trajectory.shape[0]):
        joint_pos = IK_joint_trajectory[i]
        
        transform_from_IK.append(ptr.pqs_from_transforms(dmp_gen.chain.forward(joint_pos)))
    transform_from_IK = np.array(transform_from_IK)

    np.savetxt("cartesian_trajectory_data_from_IK_trajectory.txt", transform_from_IK)

    #trajectory, trajectory_quat, IK_joint_trajectory, gripper_traj ,T = dmp_gen.compute_IK_trajectory_moveit(trajectory, trajectory_quat, gripper_traj, T ,q0 =q0, subsample_factor=5)
    """
    trajectory, IK_joint_trajectory, T = dmp_gen.compute_IK_trajectory_KDL(trajectory, T)
    """
    dmp_gen.visualize_trajectory(trajectory, IK_joint_trajectory)
    
    traj_length = min(IK_joint_trajectory.shape[0], gripper_traj.shape[0])
    # Algin length of gripper traj and generated traj
    gripper_traj = gripper_traj[:traj_length]
    IK_joint_trajectory = IK_joint_trajectory[:traj_length,:]
    
    full_trajectory = np.hstack((IK_joint_trajectory, gripper_traj.reshape(-1, 1)))
    # # Interpolate to 20Hz and Save
    interpolated_traj, interpolated_time = interpolate_joint_trajectory(full_trajectory, T, target_freq=20.0)
    save_trajectory_data(interpolated_traj, interpolated_time, "/root/catkin_ws/recordings/interpolated_traj.pkl")

    # Later, you can reload and publish it
    joint_traj, time_stamps = load_trajectory_data("/root/catkin_ws/recordings/interpolated_traj.pkl")
    
    # ROS Publishing
    
    try:
        publisher = ROSTrajectoryPublisher(['joint1', 'joint2','joint3','joint4','joint5','joint6'])
        publisher.publish_trajectory(joint_traj, time_stamps)
    except rospy.ROSInterruptException:
        print("ROS publishing interrupted.")
    