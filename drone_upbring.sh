#!/bin/bash

# Make sure any existing session is killed first
tmux kill-session -t ros_workflow 2>/dev/null || true

# Start a new session
tmux new-session -d -s ros_workflow

# Create a 2x2 grid layout
tmux split-window -h -t ros_workflow:0.0
tmux split-window -v -t ros_workflow:0.0
tmux split-window -v -t ros_workflow:0.1
# Force tiled/grid layout to ensure equal sizes
tmux select-layout -t ros_workflow tiled

# Source ROS and set exports in the first pane (roscore)
tmux send-keys -t ros_workflow:0.0 'export ROS_MASTER_URI=http://192.168.5.72:11311/' C-m
tmux send-keys -t ros_workflow:0.0 'export ROS_IP=192.168.5.72' C-m
tmux send-keys -t ros_workflow:0.0 'export ROS_HOSTNAME=192.168.5.72' C-m
tmux send-keys -t ros_workflow:0.0 'roscore' C-m

# Sleep for 5 seconds to let roscore initialize
sleep 5

# Source ROS and set exports in the second pane
tmux send-keys -t ros_workflow:0.1 'export ROS_MASTER_URI=http://192.168.5.72:11311/' C-m
tmux send-keys -t ros_workflow:0.1 'export ROS_IP=192.168.5.72' C-m
tmux send-keys -t ros_workflow:0.1 'export ROS_HOSTNAME=192.168.5.72' C-m
# Replace with your actual roslaunch command
tmux send-keys -t ros_workflow:0.1 'cd ~/rmf_ws/src/rmf_obelix' C-m
tmux send-keys -t ros_workflow:0.1 'roslaunch your_package your_launch_file.launch' C-m

# Source ROS and set exports in the third pane
tmux send-keys -t ros_workflow:0.2 'export ROS_MASTER_URI=http://192.168.5.72:11311/' C-m
tmux send-keys -t ros_workflow:0.2 'export ROS_IP=192.168.5.72' C-m
tmux send-keys -t ros_workflow:0.2 'export ROS_HOSTNAME=192.168.5.72' C-m
# Start rosbag record
tmux send-keys -t ros_workflow:0.2 'rosbag record -a -o /home/arl/bags/drone_data_$(date +%Y-%m-%d_%H-%M-%S)' C-m

# Source ROS and set exports in the fourth pane
tmux send-keys -t ros_workflow:0.3 'export ROS_MASTER_URI=http://192.168.5.72:11311/' C-m
tmux send-keys -t ros_workflow:0.3 'export ROS_IP=192.168.5.72' C-m
tmux send-keys -t ros_workflow:0.3 'export ROS_HOSTNAME=192.168.5.72' C-m
# Run commands in the fourth pane
tmux send-keys -t ros_workflow:0.3 'cd /home/arl/sceneflow_code/PD-Flow-' C-m
tmux send-keys -t ros_workflow:0.3 './pd_flowapp' C-m

# Finally, attach to the session
tmux attach-session -t ros_workflow