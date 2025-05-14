from perception import GPUObjectDetector
import rospy
from dmp_motions import ROSTrajectoryPublisher, DMPMotionGenerator, interpolate_joint_trajectory
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import pytransform3d.trajectories as ptr
import tf

class Trajectories:
    def __init__(self, urdf_path, mesh_path, pick_dmp_path, place_dmp_path, return_dmp_path):

        self.dmp_gen_pick = DMPMotionGenerator(
        urdf_path, 
        mesh_path,
        base_link="world"
        )
        """
        self.dmp_gen_place = DMPMotionGenerator(
            urdf_path, 
            mesh_path,
            base_link="world"
        )

        self.dmp_gen_return = DMPMotionGenerator(
            urdf_path, 
            mesh_path,
            base_link="world"
        )
        """
        self.chain = self.dmp_gen_pick.chain #Pick chain from any dmp object
        
        self.dmp_gen_pick.load_dmp(pick_dmp_path)
        #self.dmp_gen_place.load_dmp(place_dmp_path) 
        #self.dmp_gen_return.load_dmp(return_dmp_path)


        self.publisher = ROSTrajectoryPublisher(['joint1', 'joint2','joint3','joint4','joint5','joint6'])

    def generate_trajectory(self, motion_class, start_joint_configuration, goal_xyz):

        if motion_class == "pick":
            dmp_gen = self.dmp_gen_pick
        elif motion_class == "place":
            dmp_gen = self.dmp_gen_place
        elif motion_class == "return":
            dmp_gen = self.dmp_gen_return
            goal_xyz = dmp_gen.goal_y.copy() #Assume that return motion always moves to the original learned position.


        new_start = dmp_gen.dmp.start_y.copy()
        new_goal = dmp_gen.dmp.goal_y.copy()

        start_xyz = ptr.pqs_from_transforms(dmp_gen.chain.forward(start_joint_configuration))[0:3]

        new_start[:3] = start_xyz
        new_goal[:3] = goal_xyz

        #new_goal[3:] = np.array([0, 0, 1, 0])



        print(f"New start: {new_start}")
        print(f"New goal: {new_goal}")
        # Generate
        T, trajectory, trajectory_quat = dmp_gen.generate_trajectory(start_y=new_start, goal_y=new_goal)

        trajectory, IK_joint_trajectory, gripper_traj ,T = dmp_gen.compute_IK_trajectory(trajectory, dmp_gen.gripper_trajectory, T ,q0 =None, subsample_factor=10)


        """
        Other IK-Solvers available

        trajectory, IK_joint_trajectory, gripper_traj ,T = dmp_gen.compute_IK_trajectory_pinocchio(trajectory, gripper_traj, T, subsample_factor=2)
        trajectory, trajectory_quat, IK_joint_trajectory, gripper_traj ,T = dmp_gen.compute_IK_trajectory_moveit(trajectory, trajectory_quat, gripper_traj, T ,q0 =q0, subsample_factor=5)    
        trajectory, IK_joint_trajectory, T = dmp_gen.compute_IK_trajectory_KDL(trajectory, T)

        """

        """
        For Testing: 
        dmp_gen.visualize_trajectory(trajectory, IK_joint_trajectory)
        """
        

        """
        Old approach for aligning gripper trajectory:

        traj_length = min(IK_joint_trajectory.shape[0], gripper_traj.shape[0])

        # Algin length of gripper traj and generated traj
        gripper_traj = gripper_traj[:traj_length]
        IK_joint_trajectory = IK_joint_trajectory[:traj_length,:]
        """

        """
        New approach for gripper trajectory alignment (not sure)
        """
        gripper_traj_aligned = resample_1d_array_simplified(gripper_traj, IK_joint_trajectory.shape[0])
        
        full_trajectory = np.hstack((IK_joint_trajectory, gripper_traj_aligned.reshape(-1, 1)))
        # # Interpolate to 20Hz and Save
        interpolated_traj, interpolated_time = interpolate_joint_trajectory(full_trajectory, T, target_freq=20.0)

        """
        Maybe uncomment later for documentation

        save_trajectory_data(interpolated_traj, interpolated_time, "/root/catkin_ws/recordings/interpolated_traj.pkl")

        # Later, you can reload and publish it
        joint_traj, time_stamps = load_trajectory_data("/root/catkin_ws/recordings/interpolated_traj.pkl")
        """
        
        return interpolated_traj, interpolated_time
        
    def publish_trajectory(self, trajectory, time):
        try:
            self.publisher.publish_trajectory(trajectory, time)
        except rospy.ROSInterruptException:
            print("ROS publishing interrupted.")


def resample_1d_array_simplified(arr_1d, new_target_length):
    """
    Resamples a 1D array to a new number of points using linear interpolation.
    The original array's "shape" is preserved by interpolating values
    against their original indices.

    Args:
        arr_1d (np.ndarray or list): The original 1D array of values.
                                     It is ASSUMED that if this is a NumPy array,
                                     it is 1D, and its length is > 1.
                                     If it's a list, it's assumed to represent
                                     a 1D sequence with length > 1.
        new_target_length (int): The desired number of points (length) in the
                                 resampled array.

    Returns:
        np.ndarray: The resampled 1D array. Returns an empty float array if
                    new_target_length is 0 or negative.
    """
    # Ensure arr_1d is a NumPy array and of float type for interpolation
    # as linear interpolation will produce floats.
    if not isinstance(arr_1d, np.ndarray) or not np.issubdtype(arr_1d.dtype, np.floating):
        arr_1d = np.array(arr_1d, dtype=float)

    original_length = arr_1d.shape[0]

    # Based on user's constraint: original_length > 1 is guaranteed.
    # This means original_length is at least 2, so interp1d will have
    # at least two points to define a segment for linear interpolation.
    new_target_length = int(new_target_length)

    if new_target_length <= 0:
        return np.array([], dtype=float) # Return an empty float array

    # Create an array representing the original indices (0, 1, ..., N-1)
    original_indices = np.arange(original_length)

    # Create an array of new indices where we want to evaluate the interpolated function.
    # np.linspace correctly handles new_target_length=1 (yielding an array with the start point, e.g., [0.]).
    # It also correctly handles new_target_length = original_length.
    new_indices = np.linspace(0, original_length - 1, num=new_target_length)

    # Create the interpolation function.
    # 'kind="linear"' means we draw straight lines between the original points.
    # `bounds_error=True` (default) will raise an error if trying to interpolate outside
    # the original index range, which shouldn't happen here with linspace.
    # `fill_value` is not needed for the same reason.
    interp_func = interp1d(original_indices, arr_1d, kind='linear')

    # Apply the interpolation function to the new indices
    resampled_arr = interp_func(new_indices)

    return resampled_arr

from sensor_msgs.msg import JointState # <--- CHANGE THIS to your actual message type

# Define the topic name and message type
# You can find these using 'rostopic list' and 'rostopic info /your_topic_name' in the terminal
#TOPIC_NAME = "/open_manipulator_6dof/joint_states" # <--- CHANGE THIS to your topic name
TOPIC_NAME = "/joint_states"
MESSAGE_TYPE = JointState    # <--- CHANGE THIS to match the import above

def get_single_message():
    """
    Subscribes to a topic, waits for a single message, and returns it.
    """


    """
    ToDo: GPUDetector already initializes a node. Maybe error, because you cannot initialize 2 nodes (i think?)
    """
  
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

def get_current_joint_configuration():
    try:
        current_position_joint = get_single_message().position
        return current_position_joint
    except AttributeError:
        sys.exit(f"Message does not contain position!")

def scan_objects(listener, class_names):
    """
    ToDo:  return position and orientation of all detected objects in the frame of the gripper
           For now: Assumes, that the end-effector is static
           Maybe later: Incorporate scanning motion -> Transformation into world frame must then take into account current position of end-effector, when object is scanned.

    """
    lookup_time = rospy.Time(0)
    pose_dict = {}
    wait_duration = rospy.Duration(8.0)
    listener.waitForTransform("world", "camera_color_frame_calib", rospy.Time(0), wait_duration)
    #known_frames = listener.getFrameStrings()
    for class_name in class_names:
        try:
            
            wait_duration = rospy.Duration(2.0)
            listener.waitForTransform("world", class_name, rospy.Time(0), wait_duration)
            (translation, rotation) = listener.lookupTransform("world", class_name, lookup_time)
            print(f"{class_name}: {translation} - {rotation}")
            #print(f"{class_name}")
            #translation = np.array(translation)
            #rotation = np.array(rotation)
            #print(f"{class_name}")
            pose_dict[class_name] = translation
        except:
            pass
            
        #pose_dict[class_name] = np.concat(translation, rotation)



    return pose_dict#Example of data structure

def transform_to_worldframe(pose_e_quat, we_SE3):
    """
    Transform the pose described in the end-effector frame to a pose in the world frame.
    ToDo: Check if correct transformation.
    """

    pose_e_SE3 = ptr.transforms_from_pqs(pose_e_quat)
    pose_w_SE3 = we_SE3 @ pose_e_SE3
    return ptr.pqs_from_transforms(pose_w_SE3)

def transform_object_poses_e_to_w(dict_object_poses_e_quat, we_SE3):
    """
    ToDo: Camera is not in the center of the end-effector frame. Adjust pose_e_quat with measurements on robot (vector: end-effector -> camera)
    """
    dict_object_poses_w_quat = {}
    for object_class in dict_object_poses_e_quat:
        pose_e_quat = dict_object_poses_e_quat[object_class]
        dict_object_poses_w_quat[object_class] = transform_to_worldframe(pose_e_quat, we_SE3)
    return dict_object_poses_w_quat



if __name__ == "__main__":

    """
    Test resampling function (for gripper trajectory)

    x_sin = np.array([np.sin(x) for x in np.linspace(0,2*np.pi, 100)])
    plt.plot(x_sin)
    x_sin_new = resample_1d_array_simplified(x_sin, x_sin.shape[0]/2)
    plt.plot(x_sin_new)
    plt.show()
    """

    

    """
    ToDo: Check GPUObjectDetector:
            -which model is used/should be used? (Why is SAM an option in GPUObjectDetector?)
            -it looks like the detector broadcasts the detected positions via ROS:
                -Run GPUObjectDetector on separate Terminal and receive poses via listening to node?
                or
                -get object poses via something like detector.get_object_locations() (not implemented yet) 
    """

    rospy.init_node('internal_event_tf_listener', anonymous=True)
    listener = tf.TransformListener()
    #while True:
    #    listener = tf.TransformListener()
     #   class_names = ["box 1", "box 2", "box 3", "box 4", "box 5"]
     #   scan_objects(listener, class_names)
        
   
    """
    Init trajectory_module -> contains 3 dmps for 3 different motions ("pick", "place", "return")
    """

    urdf_path = '/root/catkin_ws/src/open_manipulator_friends/open_manipulator_6dof_description/urdf/open_manipulator_6dof.urdf'
    mesh_path = '/root/catkin_ws/src/open_manipulator_friends/open_manipulator_6dof_description/meshes'
    pick_dmp_path = '/root/catkin_ws/dmp/learned_pick_motion.pkl' #ToDo! Not yet generated
    place_dmp_path = '/root/catkin_ws/dmp/learned_place_motion.pkl' #ToDo! Not yet generated 
    return_dmp_path = '/root/catkin_ws/dmp/learned_return_motion.pkl' #ToDo! Not yet generated

    trajectory_module = Trajectories(urdf_path, mesh_path, pick_dmp_path, place_dmp_path, return_dmp_path)
    class_names = ["box 1", "box 2", "box 3", "box 4", "box 5"]

    while True:
        """
        Init to home position
        """
        #current_joint_configuration = get_current_joint_configuration()
        #trajectory_arr, time_arr = trajectory_module.generate_trajectory(self, "return", current_joint_configuration, goal_xyz)
        #trajectory_module.publish_trajectory(trajectory_arr, time_arr)

        """
        Scan for boxes in scene 
        """
        current_joint_configuration = get_current_joint_configuration()

        we_SE3 = trajectory_module.chain.forward(current_joint_configuration)#we_SE3 Homogenous transformation matrix for transforming end-effector frame in world-frame (or world-frame to end-effector frame? -> ToDo: check)
        
        dict_object_poses_w_quat = {}
        
        while not dict_object_poses_w_quat:
            dict_object_poses_w_quat = scan_objects(listener, class_names)
            print(f"{dict_object_poses_w_quat}")
        #dict_object_poses_w_quat = transform_object_poses_e_to_w(dict_object_poses_e_quat, we_SE3)

        """
        Analyze position ?
            e.g. Get feedback from scene -> if previous place motion was not succesful -> repeat picking same box and placing on previous place target
        For now:    stack box 4 on 5, then 3 on 4, then 3 on 2, then 2 on 1
                    Assume fixed orientation of the cubes: main orientation is parallel/antiparallel to x-axis (world frame), I think this works better with our dmps.
        """
        #pose_box_5 = dict_object_poses_w_quat["box 5"]
        for class_name in dict_object_poses_w_quat:
            pose_box_4 = dict_object_poses_w_quat[class_name]
            print(f"{pose_box_4}")
            break

        """
        Pick box 4
        """

        current_joint_configuration = get_current_joint_configuration()
        trajectory_arr, time_arr = trajectory_module.generate_trajectory("pick", current_joint_configuration, pose_box_4)
        trajectory_module.publish_trajectory(trajectory_arr, time_arr)
        break

        """
        ToDo:   Check if gripper is not fully closed -> pick succesfull
                If not succesful: return to home position
                else: continue
        """

        """
        Place on box 5
        """

        current_joint_configuration = get_current_joint_configuration()
        trajectory_arr, time_arr = trajectory_module.generate_trajectory(self, "place", current_joint_configuration, pose_box_5)
        trajectory_module.publish_trajectory(trajectory_arr, time_arr)

        """
        Return to home position
        """

        current_joint_configuration = get_current_joint_configuration()
        trajectory_arr, time_arr = trajectory_module.generate_trajectory(self, "return", current_joint_configuration, goal_xyz)
        trajectory_module.publish_trajectory(trajectory_arr, time_arr)

        """
        Scan for boxes in scene 
        """
        current_joint_configuration = get_current_joint_configuration()

        we_SE3 = trajectory_module.chain.forward(current_joint_configuration)#we_SE3 Homogenous transformation matrix for transforming end-effector frame in world-frame (or world-frame to end-effector frame? -> ToDo: check)
        
        dict_object_poses_e_quat = scan_objects(detector)
        dict_object_poses_w_quat = transform_object_poses_e_to_w(dict_object_poses_e_quat, we_SE3)

        """
        ToDo:   Check if box 4 is on box 5
                If not succesful:  pick box 4 -> place on box 5
                else: pick box 3 -> place on box 4

                loop
        """

















        
















    






    

    








