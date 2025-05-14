import argparse
from pathlib import Path
import os
import cv2
import numpy as np
import open3d as o3d
from scene_flow_msgs.msg import SceneFlow

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from scipy.spatial.transform import Rotation as R

# Global dictionary for topic mappings
TOPIC_MAPPING = {
    # Map both raw and processed topics to the same naming scheme
    '/raw_psi_image': {'label': '$v_0$', 'filename': 'v0', 'color': 'red'},
    '/psi_image': {'label': '$v_0$', 'filename': 'v0', 'color': 'red'},
    '/raw_h_image': {'label': '$v_1$', 'filename': 'v1', 'color': 'blue'},
    '/h_image': {'label': '$v_1$', 'filename': 'v1', 'color': 'blue'},
}

def process_timestamp(bag_path, timestamp, output_folder):
    """
    Process data at the specified timestamp from the rosbag.
    
    Args:
        bag_path (Path): Path to the rosbag file
        timestamp (float): Timestamp in seconds from the start of the bag
        output_folder (str): Folder name to save results
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create a type store
    typestore = get_typestore(Stores.ROS1_NOETIC)
    
    # Open the rosbag
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        # Get the start and end times of the bag (in nanoseconds)
        bag_start_time = reader.start_time
        bag_end_time = reader.end_time
        
        # Convert the timestamp (seconds) to nanoseconds and add to start time
        timestamp_ns = int(timestamp * 1e9)
        target_timestamp = bag_start_time + timestamp_ns
        
        # Calculate bag duration in seconds
        bag_duration_sec = (bag_end_time - bag_start_time) / 1e9
        
        # Make sure the timestamp is within the bag's time range
        if target_timestamp < bag_start_time or target_timestamp > bag_end_time:
            print(f"Warning: Requested timestamp {timestamp} seconds is outside the bag's time range.")
            print(f"Bag duration: {bag_duration_sec:.2f} seconds")
            
        print(f"Bag start time: {bag_start_time} ns ({bag_start_time/1e9:.2f} s)")
        print(f"Target timestamp: {target_timestamp} ns (bag start + {timestamp} seconds)")
        
        # Extract images
        extract_images(reader, target_timestamp, output_path)

        
        
        # Create the new control visualization
        create_control_visualization(reader, target_timestamp, output_path,
                                   time_range=3.0,         # Time range in seconds
                                   distance_threshold=3.0)  # Max distance in meters
        
        process_sceneflow(reader, target_timestamp, output_path)
        
        # Visualize drone path and point cloud
        #generate_timeline_visualization(reader, target_timestamp, output_path, 
        #                             time_range=5.0,           # Time range in seconds
        #                             point_cloud_interval=1.0,  # Sample interval
        #                             fixed_view_size=None,      # Auto-scale view
        #                             distance_threshold=3.0)    # Max distance in meters
        
        # Any other processing
        additional_processing(reader, target_timestamp, output_path)
    
    print(f"Processing complete. Results saved to {output_path}")

def extract_images(reader, timestamp, output_path):
    """
    Extract images at the specified timestamp.
    
    Args:
        reader (AnyReader): Rosbag reader
        timestamp (float): Target absolute timestamp in nanoseconds
        output_path (Path): Output directory
    """
    # Define image topics to extract
    image_topics = [
        '/color_image',
        '/psi_image',
        '/h_image',
    ]
    
    # Find connections for the image topics
    connections = [x for x in reader.connections if x.topic in image_topics]
    
    # Track the closest message for each topic
    closest_messages = {}
    
    # First pass: find the closest message for each topic
    for connection in connections:
        topic = connection.topic
        closest_diff = float('inf')
        closest_data = None
        
        for conn, msg_timestamp, rawdata in reader.messages(connections=[connection]):
            # Calculate time difference in nanoseconds
            time_diff = abs(msg_timestamp - timestamp)
            
            # Update if this message is closer to target timestamp
            if time_diff < closest_diff:
                closest_diff = time_diff
                closest_data = (conn, msg_timestamp, rawdata)
                
            # Optional optimization: if messages are time-ordered and we've passed 
            # the target timestamp by a margin, we can stop searching
            if msg_timestamp > timestamp and time_diff > 2 * closest_diff:
                break
        
        # Store the closest message data
        if closest_data:
            closest_messages[topic] = closest_data
    
    # Second pass: process only the closest message for each topic
    for topic, (conn, msg_timestamp, rawdata) in closest_messages.items():
        # Convert to seconds from bag start for display
        seconds_from_start = (msg_timestamp - reader.start_time) / 1e9
        target_seconds_from_start = (timestamp - reader.start_time) / 1e9
        time_diff_seconds = abs(msg_timestamp - timestamp) / 1e9
        
        print(f"Processing {topic}, found at {seconds_from_start:.2f}s (target: {target_seconds_from_start:.2f}s)")
        print(f"Timestamp diff: {time_diff_seconds:.6f} seconds")
        
        # Deserialize message
        msg = reader.deserialize(rawdata, conn.msgtype)
        
        # Process and save the image
        save_image(msg, topic, output_path)

def save_image(image_msg, topic_name, output_path):
    """
    Save image message to file.
    
    Args:
        image_msg: ROS image message
        topic_name (str): Topic the image came from
        output_path (Path): Output directory
    """
    # Use the mapping dictionary to get the filename
    if topic_name in TOPIC_MAPPING:
        topic_simple = TOPIC_MAPPING[topic_name]['filename']
    else:
        # Fall back to default naming for other topics
        topic_simple = topic_name.replace('/', '_').strip('_')

    # Map from ROS image encodings to OpenCV color conversion codes
    CV_CONVERSION_CODES = {
        'rgb8':    cv2.COLOR_RGB2BGR,
        'rgba8':   cv2.COLOR_RGBA2BGR,
        'rgb16':   cv2.COLOR_RGB2BGR,
        'rgba16':  cv2.COLOR_RGBA2BGR,
        'bgr8':    None,  # No conversion needed
        'bgra8':   cv2.COLOR_BGRA2BGR,
        'bgr16':   None,  # No conversion needed
        'bgra16':  cv2.COLOR_BGRA2BGR,
        'mono8':   None,  # No conversion needed for grayscale
        'mono16':  None,  # No conversion needed for grayscale
        'bayer_rggb8': cv2.COLOR_BayerBG2BGR,
        'bayer_bggr8': cv2.COLOR_BayerRG2BGR, 
        'bayer_gbrg8': cv2.COLOR_BayerGR2BGR,
        'bayer_grbg8': cv2.COLOR_BayerGB2BGR,
    }
    
    # Convert image data to numpy array
    image_data = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(image_msg.height, image_msg.width, -1)

    print(image_msg.encoding)

    if CV_CONVERSION_CODES[image_msg.encoding] is not None:
        image = cv2.cvtColor(image_data, CV_CONVERSION_CODES[image_msg.encoding])
    else:
        image = image_data
    
    # Save image
    cv2.imwrite(str(output_path / f"{topic_simple}.png"), image)
    
    print(f"Saved image from {topic_name} to {output_path / f'{topic_simple}.png'}")

def process_sceneflow(reader, timestamp, output_path):
    """
    Process sceneflow message at the specified timestamp and visualize with Open3D.
    
    Args:
        reader (AnyReader): Rosbag reader
        timestamp (float): Target timestamp
        output_path (Path): Output directory
    """
    # Define the sceneflow topic
    sceneflow_topic = '/scene_flow'
    
    # Find connection for the sceneflow topic
    connections = [x for x in reader.connections if x.topic == sceneflow_topic]
    
    if not connections:
        print(f"No sceneflow topic found matching '{sceneflow_topic}'")
        return
    
    # Find message closest to the timestamp
    closest_diff = float('inf')
    closest_data = None
    
    for conn, msg_timestamp, rawdata in reader.messages(connections=connections):
        # Calculate time difference in nanoseconds
        time_diff = abs(msg_timestamp - timestamp)
        
        # Update if this message is closer to target timestamp
        if time_diff < closest_diff:
            closest_diff = time_diff
            closest_data = (conn, msg_timestamp, rawdata)
            
        # Optional optimization: if messages are time-ordered and we've passed 
        # the target timestamp by a margin, we can stop searching
        if msg_timestamp > timestamp and time_diff > 2 * closest_diff:
            break
    
    if not closest_data:
        print(f"No sceneflow message found near the requested timestamp")
        return
        
    conn, msg_timestamp, rawdata = closest_data
    
    # Deserialize message
    msg = reader.deserialize(rawdata, conn.msgtype)
    
    # Convert to seconds from bag start for display
    seconds_from_start = (msg_timestamp - reader.start_time) / 1e9
    target_seconds_from_start = (timestamp - reader.start_time) / 1e9
    time_diff_seconds = abs(msg_timestamp - timestamp) / 1e9
    
    print(f"Processing sceneflow, found at {seconds_from_start:.2f}s (target: {target_seconds_from_start:.2f}s)")
    print(f"Timestamp diff: {time_diff_seconds:.6f} seconds")
    
    # Extract points and flow vectors from the message
    try:
        # Convert ROS message data to NumPy arrays
        pts = np.array([[pt.x, pt.y, pt.z] for pt in msg.points])
        vecs = np.array([[vec.x, vec.y, vec.z] for vec in msg.flow_vectors])
        
        print(f"Received scene flow with {pts.shape[0]} points and {vecs.shape[0]} vectors.")
        
        # Save results
        #np.save(output_path / 'sceneflow_points.npy', pts)
        #np.save(output_path / 'sceneflow_vectors.npy', vecs)
        
        # Visualize the scene flow
        visualize_sceneflow(pts, vecs, output_path)
        
    except Exception as e:
        print(f"Error processing sceneflow message: {e}")

def visualize_sceneflow(pts, vecs, output_path):
    """
    Visualize scene flow data using Open3D.
    
    Args:
        pts (np.ndarray): Point cloud data, shape (N, 3)
        vecs (np.ndarray): Flow vectors, shape (N, 3)
    """
    # Filter out points where depth (z) is less than 0.1
    depth_mask = pts[:, 2] >= 0.1
    pts_filtered = pts[depth_mask]
    vecs_filtered = vecs[depth_mask]

    # Optionally sample every Nth point/vector if the data is too dense
    pts_sampled = pts_filtered[::]
    vecs_sampled = vecs_filtered[::]

    # Create an Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_sampled)

    # Build an Open3D LineSet to represent the vectors as arrows/lines
    arrow_points = []
    arrow_lines = []
    arrow_colors = []

    # For each sampled point and its corresponding vector, add a line from the point
    # to the point plus the vector
    for p, v in zip(pts_sampled, vecs_sampled):
        start = p
        # Optionally adjust scale for visualization of the vector length
        scale = 0.2
        end = p + scale * v

        idx_start = len(arrow_points)
        arrow_points.append(start)
        idx_end = len(arrow_points)
        arrow_points.append(end)
        arrow_lines.append([idx_start, idx_end])
        # Color the arrow red
        arrow_colors.append([1.0, 0.0, 0.0])

    # Create an Open3D LineSet for the arrows
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(arrow_points))
    line_set.lines = o3d.utility.Vector2iVector(np.array(arrow_lines))
    line_set.colors = o3d.utility.Vector3dVector(np.array(arrow_colors))

    # Use the Visualizer class to enable camera control
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Scene Flow Vector Field",
        width=640,
        height=480,
        left=50,
        top=50
    )
    vis.add_geometry(pcd)
    vis.add_geometry(line_set)

    # Add a coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)
    
    # Improve rendering settings
    opt = vis.get_render_option()
    opt.point_size = 3.0
    opt.line_width = 2.0
    opt.background_color = [1.0, 1.0, 1.0]  # White background
    opt.light_on = True
    
    # Get view control
    ctr = vis.get_view_control()
    
    # Set the default camera parameters from the extrinsic matrix
    params = ctr.convert_to_pinhole_camera_parameters()
    params.extrinsic = np.array([
        [0.9954698115455077, -0.04183263560415689, 0.08538082278803111, -0.10985717201383449],
        [0.056841665625145185, 0.9817027486862883, -0.1817381035191727, 1.1129760018649744],
        [-0.07621600455619892, 0.18576798384058585, 0.9796332869136763, 2.3289899295446252],
        [0.0, 0.0, 0.0, 1.0]
    ])
    ctr.convert_from_pinhole_camera_parameters(params)
    
    # Update the renderer
    vis.poll_events()
    vis.update_renderer()
    
    print("\n------------------------------------------------------")
    print("CAMERA CONFIGURATION:")
    print("1. Default view has been applied")
    print("2. You can adjust the view using your mouse if needed")
    print("3. Close the window when ready to save the image")
    print("------------------------------------------------------\n")
    
    # Run the window to allow manual camera adjustment
    vis.run()
    
    # After the window is closed, get the final camera parameters
    params = ctr.convert_to_pinhole_camera_parameters()
    
    # Extract useful view parameters for logging
    extrinsic = params.extrinsic
    R = extrinsic[:3, :3]
    forward = -R[2, :]
    up = R[1, :]
    
    print("\n------------------------------------------------------")
    print("FINAL CAMERA CONFIGURATION:")
    print(f"# Extrinsic matrix (camera pose):")
    print(f"extrinsic_matrix = np.array({extrinsic.tolist()})")
    print(f"\n# View parameters:")
    print(f"forward = {forward.tolist()}")
    print(f"up = {up.tolist()}")
    print("------------------------------------------------------\n")
    
    # Save the image with the final view configuration
    image_path = str(output_path / 'sceneflow_visualization.png')
    vis.capture_screen_image(image_path, True)
    
    # Also save in a lossless format
    image_path_png = str(output_path / 'sceneflow_visualization_hq.png')
    vis.capture_screen_image(image_path_png, do_render=True)
    
    print(f"Saved visualization images to {image_path} and {image_path_png}")
    
    # Close the window
    vis.destroy_window()

def plot_min_image_values(reader, timestamp, output_path, time_range=5.0):
    """
    Plot the minimum values of float images around the given timestamp.
    
    Args:
        reader (AnyReader): Rosbag reader
        timestamp (float): Target timestamp in nanoseconds
        output_path (Path): Output directory
        time_range (float): Time range in seconds before and after the timestamp
    """
    # Define the image topics to track
    topics = ['/raw_h_image', '/raw_psi_image']
    
    # Find connections for the image topics
    connections = [x for x in reader.connections if x.topic in topics]
    
    if not connections:
        print(f"No image topics found matching {topics}")
        return
    
    # Define time range bounds
    start_time = timestamp - int(time_range * 1e9)  # Convert seconds to nanoseconds
    end_time = timestamp + int(time_range * 1e9)    # Convert seconds to nanoseconds
    
    # Dictionary to store results for each topic
    results = {topic: {'timestamps': [], 'min_values': []} for topic in topics}
    
    # Process messages within the time range
    for connection in connections:
        topic = connection.topic
        print(f"Processing {topic} for time series...")
        
        for conn, msg_timestamp, rawdata in reader.messages(connections=[connection]):
            # Skip messages outside our time range
            if msg_timestamp < start_time or msg_timestamp > end_time:
                continue
            
            # Deserialize message
            msg = reader.deserialize(rawdata, conn.msgtype)
            
            # Convert timestamp to seconds from the start of the bag
            seconds_from_start = (msg_timestamp - reader.start_time) / 1e9
            
            # Process image as 32-bit float
            try:
                # Assuming the data is a 32-bit float image
                float_img = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
                
                # Calculate min value
                min_value = float_img.min()
                
                # Store result
                results[topic]['timestamps'].append(seconds_from_start)
                results[topic]['min_values'].append(min_value)
                
            except Exception as e:
                print(f"Error processing {topic} at {seconds_from_start:.2f}s: {e}")
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Target timestamp in seconds from start
    target_seconds = (timestamp - reader.start_time) / 1e9
    
    # Plot data for each topic
    for topic in topics:
        if topic in TOPIC_MAPPING and results[topic]['timestamps']:
            plt.plot(
                results[topic]['timestamps'], 
                results[topic]['min_values'],
                marker='o',
                linestyle='-',
                color=TOPIC_MAPPING[topic]['color'],
                label=TOPIC_MAPPING[topic]['label']
            )
    
    # Add vertical line at target timestamp
    plt.axvline(x=target_seconds, color='green', linestyle='--', label='Target Time')
    
    # Add labels and legend
    plt.xlabel('Time (seconds from flight start)')
    plt.ylabel('Minimum Values')
    plt.title('Minimum Values Over Time ($v_0$ and $v_1$)')
    plt.legend()
    plt.grid(True)
    
    # Save the plot using new naming convention
    plot_path = output_path / 'min_values_v0_v1.pdf'
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Saved minimum value plot to {plot_path}")


def additional_processing(reader, timestamp, output_path):
    """
    Perform any additional processing at the specified timestamp.
    
    Args:
        reader (AnyReader): Rosbag reader
        timestamp (float): Target timestamp
        output_path (Path): Output directory
    """
    # Plot minimum image values over time
    plot_min_image_values(reader, timestamp, output_path, time_range=5.0)

def extract_pointcloud2_data(pc_msg):
    """Extract XYZ points from a PointCloud2 message."""
    try:
        # Get point cloud dimensions
        row_step = pc_msg.row_step
        point_step = pc_msg.point_step
        
        # Extract field offsets for x, y, z
        x_offset = y_offset = z_offset = None
        for field in pc_msg.fields:
            if field.name == 'x':
                x_offset = field.offset
            elif field.name == 'y':
                y_offset = field.offset
            elif field.name == 'z':
                z_offset = field.offset
        
        if x_offset is None or y_offset is None or z_offset is None:
            print("Could not find x, y, z fields in point cloud")
            return None
        
        # Extract point data
        points = []
        for i in range(0, len(pc_msg.data), point_step):
            x = np.frombuffer(pc_msg.data[i+x_offset:i+x_offset+4], dtype=np.float32)[0]
            y = np.frombuffer(pc_msg.data[i+y_offset:i+y_offset+4], dtype=np.float32)[0]
            z = np.frombuffer(pc_msg.data[i+z_offset:i+z_offset+4], dtype=np.float32)[0]
            
            # Skip invalid points
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                continue
                
            points.append([x, y, z])
        
        return np.array(points)
    except Exception as e:
        print(f"Error extracting point cloud data: {e}")
        return None

def downsample_pointcloud(points, voxel_size=0.1):
    """Downsample a point cloud using voxel grid filtering."""
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Downsample
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # Convert back to numpy array
    return np.asarray(downsampled_pcd.points)

def create_3d_visualization(poses, pointclouds, output_path, zoom_factor=1.5, fixed_view_size=None):
    """
    Create an enhanced 3D visualization of drone path and point clouds using matplotlib.
    
    Args:
        poses: List of drone poses
        pointclouds: List of point cloud arrays
        output_path: Path to save visualization
        zoom_factor: Factor to zoom out (>1.0 zooms out, <1.0 zooms in)
        fixed_view_size: Optional fixed size for view cube in meters
    """
    # Set a modern style that's compatible with older matplotlib versions
    try:
        # Try newer style format first
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            # Fall back to older style name
            plt.style.use('seaborn-whitegrid')
        except:
            # If all else fails, use default style
            pass
    
    # Create figure with a single 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract trajectory points
    traj_x = np.array([pose[1][0] for pose in poses])
    traj_y = np.array([pose[1][1] for pose in poses])
    traj_z = np.array([pose[1][2] for pose in poses])
    timestamps = np.array([pose[0] for pose in poses])
    
    # Normalize timestamps for colormap (0 to 1)
    if len(timestamps) > 1:
        norm_time = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
    else:
        norm_time = np.array([0.5])
    
    # Color mapping for trajectory
    cmap = plt.cm.viridis
    colors = [cmap(t) for t in norm_time]
    
    # Combine all point clouds and downsample if necessary
    combined_points = np.vstack(pointclouds)
    max_points = 5000  # Lower limit for visualization
    
    if len(combined_points) > max_points:
        # Random sampling for display
        idx = np.random.choice(len(combined_points), max_points, replace=False)
        display_points = combined_points[idx]
    else:
        display_points = combined_points
    
    # Plot path as a colored line showing progression over time
    for i in range(len(traj_x)-1):
        ax.plot(traj_x[i:i+2], traj_y[i:i+2], traj_z[i:i+2], 
               color=colors[i], linewidth=3)
    
    # Add markers for waypoints
    interval = max(1, len(traj_x) // 8)
    for i in range(0, len(traj_x), interval):
        ax.plot([traj_x[i]], [traj_y[i]], [traj_z[i]], 'o', 
               color=colors[i], markersize=8, markeredgecolor='black')
    
    # Start and end markers
    ax.plot([traj_x[0]], [traj_y[0]], [traj_z[0]], 'o', 
           color='magenta', markersize=10, markeredgecolor='black', label='Start')
    ax.plot([traj_x[-1]], [traj_y[-1]], [traj_z[-1]], 'o', 
           color='cyan', markersize=10, markeredgecolor='black', label='End')
    
    # Plot point cloud with LARGER dots
    scatter = ax.scatter(
        display_points[:, 0],
        display_points[:, 1], 
        display_points[:, 2],
        c=display_points[:, 2],  # Color by height
        cmap='terrain',
        s=15,                    # Increased point size
        alpha=0.8,               # Less transparent
        marker='o',              # Circle marker instead of point
        edgecolors='none'        # No edge color for performance
    )
    
    # Set labels and title
    ax.set_xlabel('X (m)', fontsize=12, labelpad=10)
    ax.set_ylabel('Y (m)', fontsize=12, labelpad=10)
    ax.set_zlabel('Z (m)', fontsize=12, labelpad=10)
    ax.set_title('Drone Path and Environment (3D View)', fontsize=14, pad=10)
    
    # Make axes equal for better perspective
    x_range = max(traj_x) - min(traj_x)
    y_range = max(traj_y) - min(traj_y)
    z_range = max(traj_z) - min(traj_z)
    max_range = max(x_range, y_range, z_range) / 2
    
    # Apply zoom factor to increase the view range
    max_range = max_range * zoom_factor
    
    mid_x = (max(traj_x) + min(traj_x)) / 2
    mid_y = (max(traj_y) + min(traj_y)) / 2
    mid_z = (max(traj_z) + min(traj_z)) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add grid for better depth perception
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Add colorbar for trajectory time
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(timestamps)
    time_cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.5, aspect=15)
    time_cbar.set_label('Flight Time (seconds)')
    
    # Add colorbar for point cloud height
    height_cbar = fig.colorbar(scatter, ax=ax, pad=0.02, shrink=0.5, aspect=15, location='right')
    height_cbar.set_label('Elevation (m)')
    
    # Add flight statistics
    distance = 0
    for i in range(len(traj_x)-1):
        segment = np.array([traj_x[i+1] - traj_x[i], 
                          traj_y[i+1] - traj_y[i], 
                          traj_z[i+1] - traj_z[i]])
        distance += np.linalg.norm(segment)
    
    duration = timestamps[-1] - timestamps[0]
    
    stats_text = (
        f"Flight Statistics:\n"
        f"Duration: {duration:.1f} seconds\n"
        f"Distance: {distance:.2f} meters\n"
        f"Altitude range: {min(traj_z):.1f} to {max(traj_z):.1f} m\n"
        f"Points mapped: {len(combined_points)}"
    )
    
    fig.text(0.02, 0.02, stats_text, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Save as PNG with high DPI for better quality
    plt.savefig(str(output_path / 'drone_path_3d.png'), dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path / 'drone_path_3d.pdf'), bbox_inches='tight')
    
    # Show the plot on screen
    plt.tight_layout()
    plt.show()
    
    # Close the figure after showing
    plt.close()

def create_open3d_visualization(poses, pointclouds, output_path):
    """Create an Open3D visualization of the path and point clouds."""
    # Create Open3D point cloud from all points
    combined_points = np.vstack(pointclouds)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_points)
    
    # Set point cloud color to light gray
    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    
    # Create line set for trajectory
    traj_points = np.array([pose[1] for pose in poses])
    lines = [[i, i+1] for i in range(len(traj_points)-1)]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(traj_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in range(len(lines))])  # Blue lines
    
    # Create sphere for start point (magenta)
    start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    start_sphere.translate(traj_points[0])
    start_sphere.paint_uniform_color([1, 0, 1])  # Magenta
    
    # Create sphere for end point (cyan)
    end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    end_sphere.translate(traj_points[-1])
    end_sphere.paint_uniform_color([0, 1, 1])  # Cyan
    
    # Create spheres for waypoints
    waypoint_spheres = []
    interval = max(1, len(traj_points) // 10)
    for i in range(interval, len(traj_points)-1, interval):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
        sphere.translate(traj_points[i])
        sphere.paint_uniform_color([0, 0, 1])  # Blue
        waypoint_spheres.append(sphere)
    
    # Create coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    
    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Drone Path and Environment",
        width=1024,
        height=768
    )
    
    # Add geometries
    vis.add_geometry(pcd)
    vis.add_geometry(line_set)
    vis.add_geometry(start_sphere)
    vis.add_geometry(end_sphere)
    for sphere in waypoint_spheres:
        vis.add_geometry(sphere)
    vis.add_geometry(coordinate_frame)
    
    # Rendering options
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.line_width = 5.0
    opt.background_color = [1.0, 1.0, 1.0]  # White background
    
    # Set a reasonable view
    ctr = vis.get_view_control()
    
    # Update renderer and capture image
    vis.poll_events()
    vis.update_renderer()
    
    # Allow interactive viewing if needed
    vis.run()
    
    # Save visualization image
    vis.capture_screen_image(str(output_path / 'drone_path_3d_o3d.png'), True)
    
    # Close the window
    vis.destroy_window()

def transform_points_to_global(points, position, orientation):
    """
    Transform points from base_link frame to global frame using scipy's rotation.
    
    Args:
        points (np.ndarray): Points in base_link frame, shape (N, 3)
        position (list): Global position [x, y, z]
        orientation (list): Orientation as quaternion [x, y, z, w]
    
    Returns:
        np.ndarray: Points in global frame
    """
    # Convert quaternion to scipy Rotation object
    # scipy uses [x, y, z, w] format
    qx, qy, qz, qw = orientation
    rot = R.from_quat([qx, qy, qz, qw])
    
    # Apply rotation
    rotated_points = rot.apply(points)
    
    # Apply translation
    position_array = np.array(position)
    transformed_points = rotated_points + position_array
    
    return transformed_points

def create_control_visualization(reader, timestamp, output_path, time_range=10.0, distance_threshold=5.0):
    """
    Create visualization with point cloud snapshots and control/reference data.
    
    Args:
        reader (AnyReader): Rosbag reader
        timestamp (float): Target absolute timestamp in nanoseconds
        output_path (Path): Output directory
        time_range (float): Time range in seconds to visualize
        distance_threshold (float): Maximum distance (in meters) to include points
    """
    # Define time range bounds
    start_time = timestamp - int(time_range * 1e9 / 2)
    end_time = timestamp + int(time_range * 1e9 / 2)
    
    print(f"\nCreating control visualization from {(start_time-reader.start_time)/1e9:.2f}s to {(end_time-reader.start_time)/1e9:.2f}s")
    
    # Define topics - expand the list of potential control/reference topics
    pointcloud_topic = '/depth_point_cloud'
    
    # Add more potential topics for reference and control data
    control_topics = [
        '/mavros/setpoint_raw/local_viz',    # Reference visualization
    ]
    
    reference_topics = [
        '/input_viz',                        # Primary control input visualization
    ]
    
    # Find connections
    pointcloud_conn = [x for x in reader.connections if x.topic == pointcloud_topic]
    
    # Try to find reference topic connections
    reference_conn = []
    reference_topic_found = None
    for topic in reference_topics:
        conn = [x for x in reader.connections if x.topic == topic]
        if conn:
            reference_conn = conn
            reference_topic_found = topic
            print(f"Found reference topic: {topic}")
            break
    
    # Try to find control topic connections
    control_conn = []
    control_topic_found = None
    for topic in control_topics:
        conn = [x for x in reader.connections if x.topic == topic]
        if conn:
            control_conn = conn
            control_topic_found = topic
            print(f"Found control topic: {topic}")
            break
    
    # Log all available topics if we didn't find what we need
    if not reference_conn and not control_conn:
        print("Could not find reference or control topics. Available topics:")
        for topic in sorted([conn.topic for conn in reader.connections]):
            print(f"  {topic}")
    
    # Check that we found point cloud topic
    if not pointcloud_conn:
        print(f"No point cloud topic found: {pointcloud_topic}")
        return
    
    # Plot control and reference signals over time
    if reference_conn or control_conn:
        plot_control_reference_time_series(reader, reference_conn, control_conn, 
                                          start_time, end_time, output_path)
    
    # Get 4 evenly spaced time points
    num_snapshots = 4
    time_points = []
    time_interval = time_range / (num_snapshots - 1)
    
    for i in range(num_snapshots):
        time_point = start_time + int(i * time_interval * 1e9)
        time_points.append(time_point)
    
    # Collect data for each time point
    snapshot_data = []
    
    for time_point in time_points:
        relative_time = (time_point - reader.start_time) / 1e9
        print(f"\nProcessing snapshot at {relative_time:.2f}s")
        
        # Find closest point cloud
        closest_pc = get_closest_message(reader, pointcloud_conn, time_point, max_diff=0.5 * 1e9)
        
        if not closest_pc:
            print(f"  No point cloud found near {relative_time:.2f}s")
            continue
        
        pc_time, pc_msg = closest_pc
        pc_data = extract_pointcloud2_data(pc_msg)
        
        if pc_data is None or len(pc_data) == 0:
            print(f"  Empty point cloud at {relative_time:.2f}s")
            continue
        
        # Filter by distance
        distances = np.linalg.norm(pc_data, axis=1)
        pc_data = pc_data[distances <= distance_threshold]
        
        if len(pc_data) < 10:
            print(f"  Too few points after filtering at {relative_time:.2f}s")
            continue
        
        # Get reference input
        reference_data = None
        if reference_conn:
            try:
                ref_msg = get_closest_message(reader, reference_conn, time_point)
                if ref_msg:
                    ref_time, msg = ref_msg
                    ref_time_rel = (ref_time - reader.start_time) / 1e9
                    print(f"  Found reference message at {ref_time_rel:.2f}s")
                    
                    # Try multiple message formats
                    if hasattr(msg, 'twist') and hasattr(msg.twist, 'linear'):
                        reference_data = {
                            'linear': [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z],
                            'angular': [msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z]
                        }
                        print(f"  Reference data (twist format): {reference_data}")
                    elif hasattr(msg, 'velocity'):
                        reference_data = {
                            'linear': [msg.velocity.x, msg.velocity.y, msg.velocity.z],
                            'angular': [msg.yaw_rate] if hasattr(msg, 'yaw_rate') else [0.0]
                        }
                        print(f"  Reference data (velocity format): {reference_data}")
                    elif hasattr(msg, 'linear') and hasattr(msg, 'angular'):
                        reference_data = {
                            'linear': [msg.linear.x, msg.linear.y, msg.linear.z],
                            'angular': [msg.angular.x, msg.angular.y, msg.angular.z]
                        }
                        print(f"  Reference data (direct format): {reference_data}")
                    else:
                        print(f"  Unknown reference message format: {type(msg).__name__}")
                        print(f"  Message attributes: {dir(msg)}")
            except Exception as e:
                print(f"  Error processing reference data: {e}")
        
        # Get control input
        control_data = None
        if control_conn:
            try:
                ctrl_msg = get_closest_message(reader, control_conn, time_point)
                if ctrl_msg:
                    ctrl_time, msg = ctrl_msg
                    ctrl_time_rel = (ctrl_time - reader.start_time) / 1e9
                    print(f"  Found control message at {ctrl_time_rel:.2f}s")
                    
                    # Try multiple message formats
                    if hasattr(msg, 'twist') and hasattr(msg.twist, 'linear'):
                        control_data = {
                            'linear': [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z],
                            'angular': [msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z]
                        }
                        print(f"  Control data (twist format): {control_data}")
                    elif hasattr(msg, 'control'):
                        # For actuator control messages
                        control_data = {
                            'control': msg.control[:],
                            'mode': msg.group_mix if hasattr(msg, 'group_mix') else 0
                        }
                        print(f"  Control data (actuator format): {control_data}")
                    elif hasattr(msg, 'linear') and hasattr(msg, 'angular'):
                        control_data = {
                            'linear': [msg.linear.x, msg.linear.y, msg.linear.z],
                            'angular': [msg.angular.x, msg.angular.y, msg.angular.z]
                        }
                        print(f"  Control data (direct format): {control_data}")
                    else:
                        print(f"  Unknown control message format: {type(msg).__name__}")
                        print(f"  Message attributes: {dir(msg)}")
            except Exception as e:
                print(f"  Error processing control data: {e}")
        
        # Store all data for this time point
        snapshot_data.append({
            'time': relative_time,
            'point_cloud': pc_data,
            'reference': reference_data,
            'control': control_data
        })
    
    # Create visualization if we have enough data
    if len(snapshot_data) > 0:
        print(f"\nCreating visualization with {len(snapshot_data)} snapshots")
        create_snapshot_visualization(snapshot_data, output_path)
    else:
        print("Not enough data for visualization")

def get_closest_message(reader, connections, target_time, max_diff=float('inf')):
    """
    Get the message closest to the target time.
    
    Args:
        reader (AnyReader): Rosbag reader
        connections: Message connections to search
        target_time: Target timestamp
        max_diff: Maximum allowed time difference (in ns)
        
    Returns:
        tuple: (timestamp, deserialized message) or None if not found
    """
    closest_diff = float('inf')
    closest_msg = None
    
    for conn, msg_time, rawdata in reader.messages(connections=connections):
        time_diff = abs(msg_time - target_time)
        
        if time_diff < closest_diff:
            closest_diff = time_diff
            closest_msg = (msg_time, reader.deserialize(rawdata, conn.msgtype))
    
    if closest_msg and closest_diff <= max_diff:
        return closest_msg
    return None

def create_snapshot_visualization(snapshot_data, output_path, zoom_factor=1.0):
    """
    Create individual visualizations for point clouds and control data at each timestep.
    
    Args:
        snapshot_data: List of dictionaries with time, point_cloud, reference, and control
        output_path: Path to save the visualization
        zoom_factor: Factor to zoom in (higher values = more zoomed in)
    """
    # Determine shared axis limits to keep the view consistent across all snapshots
    all_points = np.vstack([data['point_cloud'] for data in snapshot_data])
    
    # Compute maximum absolute distance from origin in any dimension
    max_abs_distance = np.max(np.abs(all_points)) * 1.2  # Add 20% margin
    
    # Apply zoom factor (divide by zoom factor to reduce the view range)
    max_abs_distance = max_abs_distance / zoom_factor
    
    # Draw optional data
    has_reference = any(data['reference'] for data in snapshot_data)
    has_control = any(data['control'] for data in snapshot_data)
    
    # Create a separate figure for each snapshot
    for i, data in enumerate(snapshot_data):
        # Create a new figure for this snapshot
        fig = plt.figure(figsize=(10, 8))
        
        # Use a clean, minimalist style
        plt.style.use('default')
        
        # Create a single 3D subplot
        ax = fig.add_subplot(111, projection='3d')
        
        # Remove the grid
        ax.grid(True)
        
        # Remove panes
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        
        # Remove z-axis and tick labels for cleaner look
        ax.zaxis.line.set_lw(0)
        #ax.set_zticks([])
        
        # Plot the point cloud
        ax.scatter(
            data['point_cloud'][:, 0],
            data['point_cloud'][:, 1],
            data['point_cloud'][:, 2],
            c='lightblue',
            s=5,
            alpha=0.6,
            marker='o',
            edgecolors='none'
        )
        
        # Plot origin (drone position)
        ax.scatter([0], [0], [0], c='red', s=100, marker='o')
        
        # Set axis limits centered around origin (0,0,0)
        #ax.set_xlim(0.5, max_abs_distance)
        #ax.set_ylim(-max_abs_distance, max_abs_distance)
        #ax.set_zlim(-max_abs_distance, max_abs_distance)

        ax.set_xlim(1, 2.5)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 0.5)
        
        # Equal aspect ratio for all axes
        ax.set_box_aspect([1, 1, 1])
        
        # Set axis labels
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        # No z label for cleaner look
        
        # Set title with time
        time_str = f"{data['time']:.2f}"
        ax.set_title(f"Time: {time_str}s", fontsize=14)
        
        # Draw reference vector if available
        if data['reference'] and 'linear' in data['reference']:
            ref = data['reference']
            # Draw vector from origin
            if len(ref['linear']) >= 3:
                # Use actual vector length instead of normalizing
                ref_vector = np.array(ref['linear'])
                if np.linalg.norm(ref_vector) > 0:
                    ax.quiver(0, 0, 0, 
                             ref_vector[0], ref_vector[1], ref_vector[2],
                             color='red', linewidth=2, label='Reference', 
                             arrow_length_ratio=0.15)
        
        # Draw control vector if available
        if data['control'] and 'linear' in data['control']:
            ctrl = data['control']
            # Draw vector from origin
            if len(ctrl['linear']) >= 3:
                # Use actual vector length instead of normalizing
                ctrl_vector = np.array(ctrl['linear'])
                if np.linalg.norm(ctrl_vector) > 0:
                    ax.quiver(0, 0, 0, 
                             ctrl_vector[0], ctrl_vector[1], ctrl_vector[2],
                             color='limegreen', linewidth=2, label='Control', 
                             arrow_length_ratio=0.15)
        
        # Add legend if we're drawing vectors
        if (data['reference'] and 'linear' in data['reference']) or (data['control'] and 'linear' in data['control']):
            ax.legend(loc='upper right')
        
        # Set the view to make y-axis more horizontal and x-axis more vertical
        ax.view_init(elev=30, azim=170)
        
        # Adjust layout
        plt.tight_layout()
        
        # Create filenames with timestamp
        png_filename = f"control_viz_t{time_str.replace('.', '_')}.png"
        pdf_filename = f"control_viz_t{time_str.replace('.', '_')}.pdf"
        
        # Save as both PNG and PDF
        dir_path = output_path / "snapshots"
        dir_path.mkdir(parents=True, exist_ok=True)
        pdf_path = dir_path / pdf_filename
        
        #plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved snapshot at t={time_str}s to {pdf_path}")
        
        # Close the figure to free memory
        plt.close(fig)
    
    print(f"Saved {len(snapshot_data)} individual snapshot images to {output_path}")

def create_3d_time_vector_visualization(reader, reference_conn, control_conn, start_time, end_time, output_path, 
                                       sample_interval=0.2, vector_scale=0.05):
    """
    Create a 3D visualization with time as a line and reference/control vectors 
    represented at intervals along the timeline.
    
    Args:
        reader (AnyReader): Rosbag reader
        reference_conn: Connection for reference messages
        control_conn: Connection for control messages
        start_time: Start timestamp in nanoseconds
        end_time: End timestamp in nanoseconds
        output_path: Path to save output plots
        sample_interval: Time interval between vector samples (seconds)
        vector_scale: Scaling factor for vector visualization (default: 0.05)
    """
    print(f"Creating 3D time-vector visualization...")
    
    # Collect reference data
    reference_samples = []
    if reference_conn:
        for conn, msg_timestamp, rawdata in reader.messages(connections=reference_conn):
            # Skip messages outside time range
            if msg_timestamp < start_time or msg_timestamp > end_time:
                continue
            
            # Convert timestamp to seconds from bag start
            rel_time = (msg_timestamp - reader.start_time) / 1e9
            
            try:
                # Deserialize message
                msg = reader.deserialize(rawdata, conn.msgtype)
                
                # Extract vector data based on message format
                if hasattr(msg, 'twist') and hasattr(msg.twist, 'linear'):
                    reference_samples.append({
                        'time': rel_time,
                        'linear': [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z],
                        'angular': [msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z]
                    })
                elif hasattr(msg, 'velocity'):
                    reference_samples.append({
                        'time': rel_time,
                        'linear': [msg.velocity.x, msg.velocity.y, msg.velocity.z],
                        'angular': [msg.yaw_rate if hasattr(msg, 'yaw_rate') else 0.0, 0, 0]
                    })
                elif hasattr(msg, 'linear') and hasattr(msg, 'angular'):
                    reference_samples.append({
                        'time': rel_time,
                        'linear': [msg.linear.x, msg.linear.y, msg.linear.z],
                        'angular': [msg.angular.x, msg.angular.y, msg.angular.z]
                    })
            except Exception as e:
                print(f"Error processing reference message: {e}")
    
    # Collect control data
    control_samples = []
    if control_conn:
        for conn, msg_timestamp, rawdata in reader.messages(connections=control_conn):
            # Skip messages outside time range
            if msg_timestamp < start_time or msg_timestamp > end_time:
                continue
            
            # Convert timestamp to seconds from bag start
            rel_time = (msg_timestamp - reader.start_time) / 1e9
            
            try:
                # Deserialize message
                msg = reader.deserialize(rawdata, conn.msgtype)
                
                # Extract vector data based on message format
                if hasattr(msg, 'twist') and hasattr(msg.twist, 'linear'):
                    control_samples.append({
                        'time': rel_time,
                        'linear': [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z],
                        'angular': [msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z]
                    })
                elif hasattr(msg, 'linear') and hasattr(msg, 'angular'):
                    control_samples.append({
                        'time': rel_time,
                        'linear': [msg.linear.x, msg.linear.y, msg.linear.z],
                        'angular': [msg.angular.x, msg.angular.y, msg.angular.z]
                    })
            except Exception as e:
                print(f"Error processing control message: {e}")
    
    # Check if we have enough data to visualize
    if not reference_samples and not control_samples:
        print("No reference or control data available for 3D visualization")
        return
    
    # Create separate visualizations for linear and angular components
    for vector_type in ['linear', 'angular']:
        # Create a new figure
        fig = plt.figure(figsize=(14, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # Time range for visualization
        rel_start = (start_time - reader.start_time) / 1e9
        rel_end = (end_time - reader.start_time) / 1e9
        
        # Draw time axis (a simple line exactly at the axis position)
        ax.plot([rel_start, rel_end], [0, 0], [0, 0], 'k-', linewidth=2)
        
        # Add time ticks without labels (we'll add custom labels)
        time_ticks = np.linspace(rel_start, rel_end, 5)
        
        # Add custom tick marks and labels
        for t in time_ticks:
            # Small vertical line for tick mark
            ax.plot([t, t], [0, -0.02], [0, 0], 'k-', linewidth=1)
            # Label below the tick
            ax.text(t, -0.04, 0, f"{t:.1f}s", color='black', fontsize=8, 
                   horizontalalignment='center', verticalalignment='top')
        
        # Sample times for vectors (evenly spaced)
        sample_times = np.arange(rel_start, rel_end, sample_interval)
        
        # Define colors for reference and control
        reference_color = 'blue'
        control_color = 'red'
        
        # Create legend handles manually
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=reference_color, lw=2, label='Reference'),
            Line2D([0], [0], color=control_color, lw=2, label='Control')
        ]
        
        # Function to find the closest sample to a given time
        def find_closest_sample(samples, target_time):
            if not samples:
                return None
            closest_sample = min(samples, key=lambda s: abs(s['time'] - target_time))
            if abs(closest_sample['time'] - target_time) > sample_interval:
                return None
            return closest_sample
        
        # Draw vectors at each sample time
        for t in sample_times:
            # Find closest reference and control samples
            ref_sample = find_closest_sample(reference_samples, t)
            ctrl_sample = find_closest_sample(control_samples, t)
            
            # Draw reference vector if available
            if ref_sample:
                vector = np.array(ref_sample[vector_type]) * vector_scale
                # Only draw if vector magnitude is significant
                if np.linalg.norm(vector) > 0.001:
                    ax.quiver(t, 0, 0,           # Origin at time point on time axis
                             0, vector[0], vector[1],  # Vector components
                             color=reference_color, arrow_length_ratio=0.3)
            
            # Draw control vector if available
            if ctrl_sample:
                vector = np.array(ctrl_sample[vector_type]) * vector_scale
                # Only draw if vector magnitude is significant
                if np.linalg.norm(vector) > 0.001:
                    ax.quiver(t, 0, 0,           # Origin at time point on time axis
                             0, vector[0], vector[1],  # Vector components
                             color=control_color, arrow_length_ratio=0.3)
        
        # Turn off all panes and grids
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        
        # Turn off grid
        ax.grid(False)
        
        # Turn off all axis lines except x-axis (time)
        ax.w_xaxis.line.set_color('black')
        ax.w_yaxis.line.set_color((0,0,0,0)) # transparent
        ax.w_zaxis.line.set_color((0,0,0,0)) # transparent
        
        # Turn off all tick marks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # No labels on any axis
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        
        # Add legend in a non-intrusive location
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Make the view horizontal to the screen (time axis horizontal)
        ax.view_init(elev=20, azim=110)
        
        # Set equal aspect ratio for the y and z dimensions
        max_range = max(abs(np.array([-0.2, 0.2])))
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        
        # Save the figure
        output_filename = f"3d_{vector_type}_control_vectors.pdf"
        plt.savefig(output_path / output_filename, bbox_inches='tight')
        print(f"Saved 3D {vector_type} vector visualization to {output_path / output_filename}")
        plt.close(fig)

def plot_control_reference_time_series(reader, reference_conn, control_conn, start_time, end_time, output_path):
    """
    Plot reference and control signals over time.
    
    Args:
        reader (AnyReader): Rosbag reader
        reference_conn: Connection for reference messages
        control_conn: Connection for control messages
        start_time: Start timestamp in nanoseconds
        end_time: End timestamp in nanoseconds
        output_path: Path to save output plots
    """
    print(f"Collecting reference and control signals for time series plot...")
    
    # Data structures to store time series
    reference_data = {
        'timestamps': [],
        'linear_x': [], 'linear_y': [], 'linear_z': [],
        'angular_x': [], 'angular_y': [], 'angular_z': []
    }
    
    control_data = {
        'timestamps': [],
        'linear_x': [], 'linear_y': [], 'linear_z': [],
        'angular_x': [], 'angular_y': [], 'angular_z': []
    }
    
    # Collect reference data
    if reference_conn:
        for conn, msg_timestamp, rawdata in reader.messages(connections=reference_conn):
            # Skip messages outside time range
            if msg_timestamp < start_time or msg_timestamp > end_time:
                continue
                
            # Convert timestamp to seconds from bag start
            rel_time = (msg_timestamp - reader.start_time) / 1e9
            
            try:
                # Deserialize message
                msg = reader.deserialize(rawdata, conn.msgtype)
                
                # Extract data based on message format
                if hasattr(msg, 'twist') and hasattr(msg.twist, 'linear'):
                    reference_data['timestamps'].append(rel_time)
                    reference_data['linear_x'].append(msg.twist.linear.x)
                    reference_data['linear_y'].append(msg.twist.linear.y)
                    reference_data['linear_z'].append(msg.twist.linear.z)
                    reference_data['angular_x'].append(msg.twist.angular.x)
                    reference_data['angular_y'].append(msg.twist.angular.y)
                    reference_data['angular_z'].append(msg.twist.angular.z)
                elif hasattr(msg, 'velocity'):
                    reference_data['timestamps'].append(rel_time)
                    reference_data['linear_x'].append(msg.velocity.x)
                    reference_data['linear_y'].append(msg.velocity.y)
                    reference_data['linear_z'].append(msg.velocity.z)
                    # Handle yaw rate if available
                    if hasattr(msg, 'yaw_rate'):
                        reference_data['angular_z'].append(msg.yaw_rate)
                    else:
                        reference_data['angular_z'].append(0.0)
                    # Fill other angular with zeros if not available
                    reference_data['angular_x'].append(0.0)
                    reference_data['angular_y'].append(0.0)
                elif hasattr(msg, 'linear') and hasattr(msg, 'angular'):
                    reference_data['timestamps'].append(rel_time)
                    reference_data['linear_x'].append(msg.linear.x)
                    reference_data['linear_y'].append(msg.linear.y)
                    reference_data['linear_z'].append(msg.linear.z)
                    reference_data['angular_x'].append(msg.angular.x)
                    reference_data['angular_y'].append(msg.angular.y)
                    reference_data['angular_z'].append(msg.angular.z)
            except Exception as e:
                print(f"Error processing reference message: {e}")
    
    # Collect control data
    if control_conn:
        for conn, msg_timestamp, rawdata in reader.messages(connections=control_conn):
            # Skip messages outside time range
            if msg_timestamp < start_time or msg_timestamp > end_time:
                continue
                
            # Convert timestamp to seconds from bag start
            rel_time = (msg_timestamp - reader.start_time) / 1e9
            
            try:
                # Deserialize message
                msg = reader.deserialize(rawdata, conn.msgtype)
                
                # Extract data based on message format
                if hasattr(msg, 'twist') and hasattr(msg.twist, 'linear'):
                    control_data['timestamps'].append(rel_time)
                    control_data['linear_x'].append(msg.twist.linear.x)
                    control_data['linear_y'].append(msg.twist.linear.y)
                    control_data['linear_z'].append(msg.twist.linear.z)
                    control_data['angular_x'].append(msg.twist.angular.x)
                    control_data['angular_y'].append(msg.twist.angular.y)
                    control_data['angular_z'].append(msg.twist.angular.z)
                elif hasattr(msg, 'linear') and hasattr(msg, 'angular'):
                    control_data['timestamps'].append(rel_time)
                    control_data['linear_x'].append(msg.linear.x)
                    control_data['linear_y'].append(msg.linear.y)
                    control_data['linear_z'].append(msg.linear.z)
                    control_data['angular_x'].append(msg.angular.x)
                    control_data['angular_y'].append(msg.angular.y)
                    control_data['angular_z'].append(msg.angular.z)
            except Exception as e:
                print(f"Error processing control message: {e}")
    
    # Check if we have enough data to plot
    if (len(reference_data['timestamps']) == 0 and len(control_data['timestamps']) == 0):
        print("No reference or control data available for time series plot")
        return
    
    print(f"Collected {len(reference_data['timestamps'])} reference points and {len(control_data['timestamps'])} control points")
    
    # Define component colors (darker for reference, lighter for control)
    component_colors = {
        'x': {'reference': '#0000AA', 'control': '#5555FF'},  # Dark and light blue
        'y': {'reference': '#AA0000', 'control': '#FF5555'},  # Dark and light red
        'z': {'reference': '#00AA00', 'control': '#55FF55'},  # Dark and light green
    }
    
    # Create separate plots for linear and angular components
    for plot_type in ['linear', 'angular']:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.tight_layout(pad=3.0)
        
        # Set a common x-axis label
        fig.text(0.5, 0.04, 'Time (seconds)', ha='center', fontsize=12)
        
        # Set a common y-axis label based on plot type
        if plot_type == 'linear':
            fig.text(0.04, 0.5, 'Control input', va='center', rotation='vertical', fontsize=12)
            fig.suptitle('Control Components Over Time', fontsize=16)
        else:
            fig.text(0.04, 0.5, 'Angular Velocity (rad/s)', va='center', rotation='vertical', fontsize=12)
            fig.suptitle('Angular Velocity Components Over Time', fontsize=16)
        
        # Plot each component (x, y, z)
        for i, axis in enumerate(['x', 'y', 'z']):
            ax = axes[i]
            
            # Reference data
            if len(reference_data['timestamps']) > 0:
                ax.plot(reference_data['timestamps'], reference_data[f'{plot_type}_{axis}'], 
                       color=component_colors[axis]['reference'], linewidth=2,
                       label=f'Reference {axis.upper()}')
            
            # Control data
            if len(control_data['timestamps']) > 0:
                ax.plot(control_data['timestamps'], control_data[f'{plot_type}_{axis}'], 
                       color=component_colors[axis]['control'], linewidth=2,
                       label=f'Control {axis.upper()}')
            
            # Add grid and legend
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
            # Set title for this component
            ax.set_title(f'{axis.upper()}-axis')
        
        # Save the figure
        output_filename = f"{plot_type}_velocity_control_reference.pdf"
        plt.savefig(output_path / output_filename, bbox_inches='tight')
        print(f"Saved {plot_type} velocity plot to {output_path / output_filename}")
        plt.close(fig)

    # Create 3D time-vector visualization
    create_3d_time_vector_visualization(reader, reference_conn, control_conn, start_time, end_time, output_path)

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Process data from a rosbag at a specific timestamp.')
    parser.add_argument('bag_path', type=str, help='Path to the rosbag file')
    parser.add_argument('timestamp', type=float, help='Timestamp in seconds from the start of the bag')
    parser.add_argument('output_folder', type=str, help='Folder name to save results')
    
    args = parser.parse_args()
    
    process_timestamp(Path(args.bag_path), args.timestamp, args.output_folder)

if __name__ == '__main__':
    main()