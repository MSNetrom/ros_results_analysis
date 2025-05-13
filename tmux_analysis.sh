#!/bin/bash

# Make sure any existing session is killed first
tmux kill-session -t res_analysis 2>/dev/null || true

# Start a new session
tmux new-session -d -s res_analysis

# Create exactly 3 panes with clear layout
tmux split-window -h
tmux split-window -v -t 1


# Source ROS in all panes
tmux send-keys -t res_analysis:0.0 'source /opt/ros/noetic/setup.bash' C-m
tmux send-keys -t res_analysis:0.1 'source /opt/ros/noetic/setup.bash' C-m
tmux send-keys -t res_analysis:0.2 'source /opt/ros/noetic/setup.bash' C-m

# Start roscore
tmux send-keys -t res_analysis:0.2 'roscore' C-m

# Send rosbag command to the first pane
tmux send-keys -t res_analysis:0.0 'rosbag play -l ../sceneflow_bags/'

# Send command to second pane
tmux send-keys -t res_analysis:0.1 'rosrun rviz rviz -d sceneflow_cbf.rviz'

# Finally, attach to the session and pane 0
tmux attach-session -t res_analysis
tmux select-pane -t 1