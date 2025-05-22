# Make sure any existing session is killed first
tmux kill-session -t ros_workflow 2>/dev/null || true

# Start a new session
tmux new-session -d -s ros_workflow

# Create a 2x2 grid layout
tmux split-window -h -t ros_workflow:0.0
# Force tiled/grid layout to ensure equal sizes
#tmux select-layout -t ros_workflow tiled

# Source ROS and set exports in the first pane (roscore)
tmux send-keys -t ros_workflow:0.0 'export ROS_MASTER_URI=http://192.168.5.72:11311/' C-m
tmux send-keys -t ros_workflow:0.0 'export ROS_IP=192.168.5.127' C-m
tmux send-keys -t ros_workflow:0.0 'export ROS_HOSTNAME=192.168.5.127' C-m
tmux send-keys -t ros_workflow:0.0 'roslaunch joystick_control joystick_control.launch' C-m

# Source ROS and start rviz
tmux send-keys -t ros_workflow:0.1 'export ROS_MASTER_URI=http://192.168.5.72:11311/' C-m
tmux send-keys -t ros_workflow:0.1 'export ROS_IP=192.168.5.127' C-m
tmux send-keys -t ros_workflow:0.1 'export ROS_HOSTNAME=192.168.5.127' C-m
tmux send-keys -t ros_workflow:0.1 'rosrun rviz rviz -d sceneflow_cbf.rviz' C-m

# Attach to the session
tmux attach-session -t ros_workflow
