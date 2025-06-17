# Use Ubuntu 20.04 as the base image
FROM ubuntu:20.04

# Set up the environment to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install ROS Noetic
RUN apt-get update && apt-get install -y curl gnupg2 lsb-release
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list

# Update the package list and install essential packages
RUN apt-get update && apt-get install -y \
    wget \
    ca-certificates \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libx11-dev \
    libxrender-dev \
    libxrandr-dev \
    build-essential \
    udev \
    usbutils \
    libusb-1.0-0-dev \
    xserver-xorg \
    x11-xserver-utils \
    x11-utils \
    mesa-utils \
    pkg-config \
    libxau-dev \
    libxdmcp-dev \
    libxcb1-dev \
    libxext-dev \
    git\
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2

# Create required directories for NVIDIA
RUN mkdir -p /usr/local/share/glvnd/egl_vendor.d/
RUN echo '{"file_format_version" : "1.0.0", "ICD" : {"library_path" : "libEGL_nvidia.so.0"}}' > /usr/local/share/glvnd/egl_vendor.d/10_nvidia.json

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics


RUN apt-get install -y \
    ros-noetic-desktop-full\
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    python3-catkin-tools \
    ros-noetic-gazebo-ros-pkgs \
    ros-noetic-gazebo-ros-control \
    ros-noetic-rviz \
    ros-noetic-ros-controllers \
    gazebo11 \
    ros-noetic-moveit \
    ros-noetic-industrial-core \
    ros-noetic-dynamixel-sdk \
    ros-noetic-dynamixel-workbench \
    ros-noetic-robotis-manipulator \
    ros-noetic-pcl-ros \
    ros-noetic-joy \
    ros-noetic-control-toolbox \
    ros-noetic-controller-interface \
    ros-noetic-controller-manager \
    ros-noetic-joint-state-controller \
    libboost-dev \
    libeigen3-dev \
    libtinyxml-dev \
    libpcl-dev\
    usb-modeswitch \
    ros-noetic-moveit-visual-tools\
    python3-pip \
    cron

# Initialize rosdep
RUN rosdep init && rosdep update

# Setup ROS workspace
RUN mkdir -p /root/catkin_ws/src && \
    cd /root/catkin_ws/src && \
    git clone https://github.com/shailjadav/ARL_25_noetic_packages.git .

# Fix: Source ROS setup before building
WORKDIR /root/catkin_ws
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# Install any missing dependencies
RUN rosdep install --from-paths src --ignore-src -y || true


# Add user for accessing USB devices
RUN groupadd -r docker && usermod -aG docker root

# Try building everything except problematic packages
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    source /root/catkin_ws/devel/setup.bash && \
    catkin_make -DCATKIN_BLACKLIST_PACKAGES=\"open_manipulator_6dof_teleop;joystick_drivers\""

# Set up environment for running GUI applications
ENV QT_X11_NO_MITSHM=1
ENV DISPLAY=:0
RUN apt-get install -y x11-apps && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/IntelRealSense/librealsense/raw/master/scripts/libuvc_installation.sh
RUN chmod +x ./libuvc_installation.sh
RUN ./libuvc_installation.sh

RUN apt-get install -y ros-noetic-realsense2-camera vim

# Source the ROS environment by default
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
RUN echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc

# Install extra python packages
RUN pip install pytransform3d
RUN pip install movement_primitives
RUN pip install numba
RUN pip install h5py
# RUN pip install -r /root/catkin_ws/src/my_scripts/requirements.txt
RUN pip install open3d
RUN curl -fsSL https://ollama.com/install.sh | sh
RUN pip install langchain_ollama
RUN git clone https://github.com/sinedo/TowerOfHanoi.git

# create symlinks
RUN cd src/om_position_controller/ && ln -s ~/TowerOfHanoi TowerOfHanoi
RUN ollama serve & exec ollama pull gemma3:12b
# RUN ln -s src/om_position_controller/scripts/4_rs_detect_sim.py .

RUN echo "roscd om_position_controller/scripts"

# Set the default command
CMD [ "bash", "-c", "ollama serve & exec bash" ]

# ENTRYPOINT [ "bash" ]