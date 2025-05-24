from collections import defaultdict
import argparse
import yaml
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union, Tuple
import bisect
from dataclasses import dataclass
from abc import ABC, abstractmethod
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D


COLORS = {
    "u_safe": "green",
    "u_filtered": "blue",
    "u_actual": "orange",
    "u_ref": "purple",
}

def convert_seconds_to_bag_time_in_nanoseconds(start_time: int, seconds: float):
    return start_time + int(seconds * 1e9)

def convert_bag_time_in_nanoseconds_to_seconds(start_time: int, seconds: int):
    """
    Start time is in nanoseconds, seconds is in seconds
    """
    return seconds - start_time / 1e9

@dataclass
class ProcessorConfig:
    enabled: bool
    type: str
    params: Dict[str, Any]
    input_mappings: Dict[str, str]  # Logical -> Actual topic

@dataclass
class BagProcessingConfig:
    def __init__(self, config_path: Path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        bp = config['bag_processing']
        self.bag_path = Path(bp['bag_path'])
        self.interval: Tuple[float, float, List[float]] = (
            bp['interval']['start'],
            bp['interval']['end'],
            bp['interval'].get('highlights', [])
        )
        self.output_dir = Path(bp['output_dir'])
        self.ros_version = bp['ros_version']
        self.max_time_diff_ns = int(bp['max_time_diff'] * 1e9)
        
        self.processors = {}
        for name, cfg in config['processors'].items():
            self.processors[name] = ProcessorConfig(
                enabled=cfg.get('enabled', True),
                type=cfg['type'],
                params=cfg.get('params', {}),
                input_mappings=cfg.get('input_mappings', {})
            )

    def validate(self):
        if self.interval[0] >= self.interval[1]:
            raise ValueError("Interval start must be before end")
        if any(not self.interval[0] <= h <= self.interval[1] 
               for h in self.interval[2]):
            raise ValueError("All highlights must be within interval bounds")

class DataExtractor:

    def __init__(self, config: BagProcessingConfig):
        self.config = config
        self.typestore = get_typestore(Stores.ROS1_NOETIC if 
            config.ros_version == "ROS1" else Stores.LATEST)
        

    def extract(self) -> Dict[str, Dict[str, Union[List[Any], List[int]]]]:

        extracted = defaultdict(lambda: {'continuous': [], 'highlights': []})
        required_topics = set()
        
        for proc in self.config.processors.values():
            if proc.enabled:
                required_topics.update(proc.input_mappings.values())

        with AnyReader([self.config.bag_path]) as reader:
            bag_start = reader.start_time
            start_ns = convert_seconds_to_bag_time_in_nanoseconds(bag_start, self.config.interval[0])
            end_ns = convert_seconds_to_bag_time_in_nanoseconds(bag_start, self.config.interval[1])
            highlight_ns = [convert_seconds_to_bag_time_in_nanoseconds(bag_start, ts) for ts in self.config.interval[2]]

            for topic in required_topics:

                extracted[topic]['bag_start'] = bag_start

                conns = [c for c in reader.connections if c.topic == topic]
                if not conns:
                    continue

                # Extract continuous data
                messages = []
                times = []
                for conn, timestamp, rawdata in reader.messages(connections=conns):
                    if start_ns <= timestamp <= end_ns:
                        msg = reader.deserialize(rawdata, conn.msgtype)
                        messages.append(msg)
                        times.append(timestamp)
                
                extracted[topic]['continuous'] = messages
                
                # Find highlight indices
                highlight_indices = []
                for hn in highlight_ns:
                    idx = bisect.bisect_left(times, hn)
                    candidates = []
                    if idx > 0:
                        candidates.append((idx-1, abs(times[idx-1] - hn)))
                    if idx < len(times):
                        candidates.append((idx, abs(times[idx] - hn)))
                    
                    if candidates:
                        best_idx, diff = min(candidates, key=lambda x: x[1])
                        if diff <= self.config.max_time_diff_ns:
                            # Print the time of the message
                            highlight_indices.append(best_idx)
                
                extracted[topic]['highlights'] = highlight_indices

                extracted[topic]['timestamps'] = times

        return extracted

class Processor(ABC):

    def __init__(self, params: Dict):
        self.params = params

    def get_timestamps_of_message_list(self, messages: List[Any], bag_start: int, zero_based: bool = True):
        timestamps = []
        for msg in messages:
            timestamps.append(convert_bag_time_in_nanoseconds_to_seconds(bag_start, msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9))

        timestamps = np.array(timestamps)

        if zero_based:
            timestamps -= timestamps[0]

        return timestamps

    @abstractmethod
    def process(self,
               data: Dict[str, Dict[str, Union[List[Any], List[int]]]],
               input_mappings: Dict[str, str],
               output_dir: Path):
        
        output_dir.mkdir(parents=True, exist_ok=True)

class ImageProcessor(Processor):

    def __init__(self, params: Dict):
        self.params = params
        self.conversions = {
            'rgb8': cv2.COLOR_RGB2BGR,
            'rgba8': cv2.COLOR_RGBA2BGR,
            'bgr8': None,
            'bgra8': cv2.COLOR_BGRA2BGR
        }


    def process(self, data, input_mappings, output_dir):
        
        super().process(data, input_mappings, output_dir)

        topic = input_mappings.get('color_image')

        if not topic or topic not in data:
            return

        # Process highlighted frames
        for idx in data[topic]['highlights']:
            msg = data[topic]['continuous'][idx]
            ts = convert_bag_time_in_nanoseconds_to_seconds(data[topic]['bag_start'], msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9)
            self._save_image(msg, ts, output_dir)

    def _save_image(self, msg, timestamp, output_dir):
        
        filename = f"{self.params['filename_prefix']}{timestamp:.2f}.png"
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
        
        if msg.encoding in self.conversions and self.conversions[msg.encoding]:
            img = cv2.cvtColor(img, self.conversions[msg.encoding])

        output_dir = output_dir / (self.params['filename_prefix'] + "color_images")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(output_dir / filename), img)

class MinCBFPlotProcessor(Processor):

    def process(self, data, input_mappings, output_dir):
        super().process(data, input_mappings, output_dir)

        topic_v_0 = input_mappings.get('v_0')
        topic_v_1 = input_mappings.get('v_1')

        if not topic_v_0 or not topic_v_1 or topic_v_0 not in data or topic_v_1 not in data:
            return
        
        v_0_messages = data[topic_v_0]['continuous']
        v_1_messages = data[topic_v_1]['continuous']

        # Convert image messages to numpy arrays and extract scalar values
        v_0_values = []
        v_1_values = []
        
        for msg in v_0_messages:
            float_img = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            # Extract scalar value from image (modify this based on your needs)
            scalar_value = np.min(float_img)  # or np.mean(img), img[0,0], etc.
            v_0_values.append(scalar_value)
            
        
        for msg in v_1_messages:
            float_img = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            # Extract scalar value from image (modify this based on your needs)
            scalar_value = np.min(float_img)  # or np.mean(img), img[0,0], etc.
            v_1_values.append(scalar_value)

        # Make timestamps relative to start of interval
        timestamps = self.get_timestamps_of_message_list(v_0_messages, data[topic_v_0]['bag_start'])

        # Make plot from all timestamps in range
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, v_0_values, label="$v_0$", linewidth=2)
        plt.plot(timestamps, v_1_values, label="$v_1$", linewidth=2)
        plt.legend(fontsize=14)
        plt.xlabel("Time [s]", fontsize=14)
        plt.ylabel("Value", fontsize=14)
        plt.title("Min CBF Plot", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)

        plt.savefig(output_dir / "min_cbf_plot.pdf")
        plt.close()

class USafetyErrorPlotProcessor(Processor):

    def _extract_values_from_message(self, twist_stamped_list):

        values = []
        for msg in twist_stamped_list:
            values.append([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
        return np.array(values)

    def process(self, data, input_mappings, output_dir):
        super().process(data, input_mappings, output_dir)

        u_safe = data[input_mappings['u_safe']]['continuous']
        u_filtered = data[input_mappings['u_filtered']]['continuous']
        u_actual = data[input_mappings['u_actual']]['continuous']

        u_safe_values = self._extract_values_from_message(u_safe)
        u_filtered_values = self._extract_values_from_message(u_filtered)
        u_actual_values = self._extract_values_from_message(u_actual)
        timestamps = self.get_timestamps_of_message_list(u_safe, data[input_mappings['u_safe']]['bag_start'])

        # Get distance from safe to the others
        distances_filtered = np.linalg.norm(u_safe_values - u_filtered_values, axis=1)
        distances_actual = np.linalg.norm(u_safe_values - u_actual_values, axis=1)

        # Make plot from all timestamps in range
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, distances_filtered, label="Filtered", color=COLORS["u_filtered"], linewidth=2)
        plt.plot(timestamps, distances_actual, label="Final (clamped)", color=COLORS["u_actual"], linewidth=2)
        plt.legend(fontsize=14)
        plt.xlabel("Time [s]", fontsize=14)
        plt.ylabel("Distance [$m/s^2$]", fontsize=14)
        plt.title("Difference between safe and filtered/final control vector", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)

        plt.savefig(output_dir / "u_safety_error_plot.pdf")
        plt.close()

class URefErrorPlotProcessor(USafetyErrorPlotProcessor):

    def process(self, data, input_mappings, output_dir):
        Processor.process(self, data, input_mappings, output_dir)

        u_ref = data[input_mappings['u_ref']]['continuous']
        u_safe = data[input_mappings['u_safe']]['continuous']
        u_filtered = data[input_mappings['u_filtered']]['continuous']
        u_actual = data[input_mappings['u_actual']]['continuous']

        u_ref_values = self._extract_values_from_message(u_ref)
        u_safe_values = self._extract_values_from_message(u_safe)
        u_filtered_values = self._extract_values_from_message(u_filtered)
        u_actual_values = self._extract_values_from_message(u_actual)

        distance_safe = np.linalg.norm(u_ref_values - u_safe_values, axis=1)
        distance_filtered = np.linalg.norm(u_ref_values - u_filtered_values, axis=1)
        distance_actual = np.linalg.norm(u_ref_values - u_actual_values, axis=1)

        timestamps = self.get_timestamps_of_message_list(u_ref, data[input_mappings['u_ref']]['bag_start'])

        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, distance_safe, label="Safe", color=COLORS["u_safe"], linewidth=2)
        plt.plot(timestamps, distance_filtered, label="Filtered", color=COLORS["u_filtered"], linewidth=2)
        plt.plot(timestamps, distance_actual, label="Final (clamped)", color=COLORS["u_actual"], linewidth=2)
        plt.legend(fontsize=14)
        plt.xlabel("Time [s]", fontsize=14)
        plt.ylabel("Distance [$m/s^2$]", fontsize=14)
        plt.title("Difference between reference and safe/filtered/final control vector", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)

        plt.savefig(output_dir / "u_ref_error_plot.pdf")
        plt.close()
        
class USizesPlotProcessor(USafetyErrorPlotProcessor):

    def process(self, data, input_mappings, output_dir):
        Processor.process(self, data, input_mappings, output_dir)

        u_ref = data[input_mappings['u_ref']]['continuous']
        u_safe = data[input_mappings['u_safe']]['continuous']
        u_filtered = data[input_mappings['u_filtered']]['continuous']
        u_actual = data[input_mappings['u_actual']]['continuous']

        u_ref_sizes = np.linalg.norm(self._extract_values_from_message(u_ref), axis=1)
        u_safe_sizes = np.linalg.norm(self._extract_values_from_message(u_safe), axis=1)
        u_filtered_sizes = np.linalg.norm(self._extract_values_from_message(u_filtered), axis=1)
        u_actual_sizes = np.linalg.norm(self._extract_values_from_message(u_actual), axis=1)

        timestamps = self.get_timestamps_of_message_list(u_ref, data[input_mappings['u_ref']]['bag_start'])

        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, u_safe_sizes, label="Safe", color=COLORS["u_safe"], linewidth=2)
        plt.plot(timestamps, u_filtered_sizes, label="Filtered", color=COLORS["u_filtered"], linewidth=2)
        plt.plot(timestamps, u_actual_sizes, label="Final (clamped)", color=COLORS["u_actual"], linewidth=2)
        plt.plot(timestamps, u_ref_sizes, label="Reference", linestyle='--', linewidth=4, color=COLORS["u_ref"])
        plt.legend(fontsize=14)

        plt.xlabel("Time [s]", fontsize=14)
        plt.ylabel("Size [$m/s^2$]", fontsize=14)
        plt.title("Size of control vectors", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)

        plt.savefig(output_dir / "u_sizes_plot.pdf")
        plt.close()


class VelocitySizePlotProcessor(Processor):

    def process(self, data, input_mappings, output_dir):
        super().process(data, input_mappings, output_dir)

        topic = input_mappings.get('odometry')
        
        if not topic or topic not in data:
            return
        
        odometry_messages = data[topic]['continuous']
        
        # Extract linear velocity values from odometry messages
        velocity_values = []
        for msg in odometry_messages:
            linear_vel = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
            velocity_values.append(linear_vel)
        
        velocity_values = np.array(velocity_values)
        
        # Calculate velocity magnitudes (sizes)
        velocity_sizes = np.linalg.norm(velocity_values, axis=1)
        
        # Get timestamps relative to start
        timestamps = self.get_timestamps_of_message_list(odometry_messages, data[topic]['bag_start'])
        
        # Create velocity magnitude plot
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, velocity_sizes, label="Velocity magnitude", linewidth=2)
        plt.legend(fontsize=14)
        plt.xlabel("Time [s]", fontsize=14)
        plt.ylabel("Velocity [m/s]", fontsize=14)
        plt.title("Robot velocity magnitude over time", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        
        plt.savefig(output_dir / "velocity_size_plot.pdf")
        plt.close()

class SnapshotVisualizationProcessor(Processor):

    def __init__(self, params: Dict):
        super().__init__(params)

    def _extract_sceneflow_data(self, msg):
        """Extract points and flow vectors from sceneflow message"""
        pts = np.array([[pt.x, pt.y, pt.z] for pt in msg.points])
        vecs = np.array([[vec.x, vec.y, vec.z] for vec in msg.flow_vectors])
        
        # Filter out invalid points and vectors
        valid_mask = ~(np.isnan(pts).any(axis=1) | np.isinf(pts).any(axis=1) | 
                      np.isnan(vecs).any(axis=1) | np.isinf(vecs).any(axis=1))
        
        # Filter out points that are too far away
        valid_mask = (np.linalg.norm(pts, axis=1) < self.params.get('max_distance', 100)) & valid_mask
        
        return pts[valid_mask], vecs[valid_mask]

    def _extract_control_vector(self, msg):
        """Extract control vector from TwistStamped message"""
        return np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])

    def _create_3d_plot(self, points, flow_vectors, u_ref_vec, u_safe_vec, u_filtered_vec, u_actual_vec, timestamp, output_dir, filename_prefix):
        """Create and save 3D visualization with sceneflow"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        print("Creating 3D plot")
        
        # Configure plot appearance
        ax.grid(True)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        
        # Plot point cloud
        if len(points) > 0:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                      c='lightblue', s=5, alpha=0.6, marker='o', edgecolors='none')
        
        # Plot flow vectors
        if len(flow_vectors) > 0:
            # Subsample flow vectors for clarity if there are too many
            max_vectors = self.params.get('max_flow_vectors', 5000000)
            if len(flow_vectors) > max_vectors:
                indices = np.linspace(0, len(flow_vectors)-1, max_vectors, dtype=int)
                plot_points = points[indices]
                plot_vectors = flow_vectors[indices]
            else:
                plot_points = points
                plot_vectors = flow_vectors
            
            # Scale flow vectors for visibility
            vector_scale = self.params.get('flow_vector_scale', 1.0)
            scaled_vectors = plot_vectors * vector_scale
            
            ax.quiver(plot_points[:, 0], plot_points[:, 1], plot_points[:, 2],
                     scaled_vectors[:, 0], scaled_vectors[:, 1], scaled_vectors[:, 2],
                     color='red', alpha=0.5, linewidth=1.0, arrow_length_ratio=0.01)
        
        # Plot origin (drone position)
        ax.scatter([0], [0], [0], c='blue', s=100, marker='o')

        # Get u-scaling factor
        u_scaling_factor = self.params.get('u_scaling_factor', 1.0)

        u_ref_vec = u_ref_vec * u_scaling_factor
        u_filtered_vec = u_filtered_vec * u_scaling_factor
        u_safe_vec = u_safe_vec * u_scaling_factor
        u_actual_vec = u_actual_vec * u_scaling_factor

        if u_ref_vec is not None:
            # Plot reference vector if available
            ax.quiver(0, 0, 0, u_ref_vec[0], u_ref_vec[1], u_ref_vec[2],
                    color=COLORS["u_ref"], linewidth=2, label=f'$\| u_{{ref}} \| = {np.linalg.norm(u_ref_vec):.1f} m/s^2$ (Scale: {u_scaling_factor:.1f})', arrow_length_ratio=0.15)
        
        if u_filtered_vec is not None:
            # Plot control vector if available
            ax.quiver(0, 0, 0, u_filtered_vec[0], u_filtered_vec[1], u_filtered_vec[2],
                color=COLORS["u_filtered"], linewidth=2, label=f'$\| u_{{filtered}} \| = {np.linalg.norm(u_filtered_vec):.1f} m/s^2$ (Scale: {u_scaling_factor:.1f})', arrow_length_ratio=0.15)
        
        if u_safe_vec is not None:
            ax.quiver(0, 0, 0, u_safe_vec[0], u_safe_vec[1], u_safe_vec[2],
                color=COLORS["u_safe"], linewidth=2, label=f'$\| u_{{safe}} \| = {np.linalg.norm(u_safe_vec):.1f} m/s^2$ (Scale: {u_scaling_factor:.1f})', arrow_length_ratio=0.15)
        
        if u_actual_vec is not None:
            ax.quiver(0, 0, 0, u_actual_vec[0], u_actual_vec[1], u_actual_vec[2],
                color=COLORS["u_actual"], linewidth=2, label=f'$\| u_{{actual}} \| = {np.linalg.norm(u_actual_vec):.1f} m/s^2$ (Scale: {u_scaling_factor:.1f})', arrow_length_ratio=0.15)
        
        # Set axis limits based on parameters
        limits = self.params.get('axis_limits', {'x': [1, 2.5], 'y': [-1, 1], 'z': [-1, 0.5]})
        ax.set_xlim(limits['x'])
        ax.set_ylim(limits['y'])
        ax.set_zlim(limits['z'])
        
        # Configure appearance
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel('X (m)', fontsize=14)
        ax.set_title(f"Sceneflow at Time: {timestamp:.2f}s", fontsize=14)
        
        # Set tick label font sizes
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='z', labelsize=12)
        
        # Set view angle
        view_params = self.params.get('view_angle', {'elev': 0, 'azim': 0, 'roll': 0})
        ax.view_init(elev=view_params['elev'], azim=view_params['azim'], roll=view_params['roll'])
        
        if u_actual_vec is not None or u_safe_vec is not None or u_filtered_vec is not None or u_ref_vec is not None:
            # Add legend if vectors are present
            legend = ax.legend(loc='upper right', bbox_to_anchor=(0.8, 0.8), fontsize=14)
            legend.set_zorder(1000)  # Ensure legend is in front of everything
        
        plt.tight_layout()
        
        # Save plot
        snapshots_dir = output_dir / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{filename_prefix}sceneflow_viz_t{timestamp:.2f}".replace('.', '_') + ".pdf"
        plt.savefig(snapshots_dir / filename, bbox_inches='tight')
        plt.close(fig)

    def process(self, data, input_mappings, output_dir):
        super().process(data, input_mappings, output_dir)
        
        # Get required topics
        sceneflow_topic = input_mappings.get('sceneflow')  # Changed from 'point_cloud'
        u_ref_topic = input_mappings.get('u_ref', None)
        u_safe_topic = input_mappings.get('u_safe', None)
        u_filtered_topic = input_mappings.get('u_filtered', None)
        u_actual_topic = input_mappings.get('u_actual', None)
        
        if not sceneflow_topic or sceneflow_topic not in data:
            print(f"Warning: Sceneflow topic {sceneflow_topic} not found")
            return
        
        # Process each highlight timestamp
        sceneflow_highlights = data[sceneflow_topic]['highlights']
        sceneflow_messages = data[sceneflow_topic]['continuous']
        
        for idx in sceneflow_highlights:

            if idx >= len(sceneflow_messages):
                continue
                
            sceneflow_msg = sceneflow_messages[idx]
            timestamp = convert_bag_time_in_nanoseconds_to_seconds(
                data[sceneflow_topic]['bag_start'], 
                data[sceneflow_topic]['timestamps'][idx] / 1e9
            )
            
            # Extract sceneflow data
            try:
                points, flow_vectors = self._extract_sceneflow_data(sceneflow_msg)
            except Exception as e:
                print(f"Error extracting sceneflow at t={timestamp:.2f}: {e}")
                continue
            
            # Extract reference vector if available
            u_ref_vec = self._extract_control_vector(data[u_ref_topic]['continuous'][idx]) if u_ref_topic is not None else None
            u_safe_vec = self._extract_control_vector(data[u_safe_topic]['continuous'][idx]) if u_safe_topic is not None else None
            u_filtered_vec = self._extract_control_vector(data[u_filtered_topic]['continuous'][idx]) if u_filtered_topic is not None else None
            u_actual_vec = self._extract_control_vector(data[u_actual_topic]['continuous'][idx]) if u_actual_topic is not None else None
            
            # Create 3D visualization
            self._create_3d_plot(points, flow_vectors, u_ref_vec, u_safe_vec, u_filtered_vec, u_actual_vec, 
                                 timestamp, output_dir, self.params.get('filename_prefix', ''))


class CBFValueImageProcessor(Processor):

    def process(self, data, input_mappings, output_dir):
        super().process(data, input_mappings, output_dir)

        v_0_dir = output_dir / "v_0_images"
        v_1_dir = output_dir / "v_1_images"
        v_0_dir.mkdir(parents=True, exist_ok=True)
        v_1_dir.mkdir(parents=True, exist_ok=True)

        v_0_data = data[input_mappings['v_0']]
        v_1_data = data[input_mappings['v_1']]
        timestamps = self.get_timestamps_of_message_list(v_0_data['continuous'], v_0_data['bag_start'])

        for idx in v_0_data['highlights']:
            v_0_np = np.frombuffer(v_0_data['continuous'][idx].data, dtype=np.uint8).reshape(v_0_data['continuous'][idx].height, v_0_data['continuous'][idx].width, -1)
            v_1_np = np.frombuffer(v_1_data['continuous'][idx].data, dtype=np.uint8).reshape(v_1_data['continuous'][idx].height, v_1_data['continuous'][idx].width, -1)

            cv2.imwrite(str(v_0_dir / f"v_0_image_{timestamps[idx]:.2f}.png"), v_0_np)
            cv2.imwrite(str(v_1_dir / f"v_1_image_{timestamps[idx]:.2f}.png"), v_1_np)
        

class TimeSeriesVectorVisualizationProcessor(Processor):

    def __init__(self, params: Dict):
        super().__init__(params)

    def _find_closest_message_index(self, timestamps, target_time_ns, max_diff_ns):
        """Find the index of the message closest to target time"""
        if not timestamps:
            return None
        
        # Use bisect to find insertion point
        idx = bisect.bisect_left(timestamps, target_time_ns)
        candidates = []
        
        if idx > 0:
            candidates.append((idx-1, abs(timestamps[idx-1] - target_time_ns)))
        if idx < len(timestamps):
            candidates.append((idx, abs(timestamps[idx] - target_time_ns)))
        
        if not candidates:
            return None
            
        best_idx, diff = min(candidates, key=lambda x: x[1])
        return best_idx if diff <= max_diff_ns else None

    def _extract_control_vector(self, msg):
        """Extract control vector from TwistStamped message"""
        return np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])

    def _create_time_series_plot(self, vector_data, output_dir):
        """Create 3D time-series vector plot"""
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        print("Creating time-series vector plot")
        
        # Configure plot appearance
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        ax.grid(False)
        
        # Hide the 3D axis lines completely
        ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        
        # Get time range
        times = sorted(vector_data.keys())
        if not times:
            return
            
        time_start, time_end = min(times), max(times)
        
        # Draw time axis
        ax.plot([time_start, time_end], [0, 0], [0, 0], 'k-', linewidth=3, alpha=0.8)
        
        # Add time ticks and labels
        num_ticks = self.params.get('num_time_ticks', 5)
        time_ticks = np.linspace(time_start, time_end, num_ticks)
        
        for t in time_ticks:
            # Small vertical line for tick mark
            ax.plot([t, t], [0, -0.05], [0, 0], 'k-', linewidth=1)

            # Label below the tick
            ax.text(t, -0.1, 0, f"{t-time_start:.2f}s", color='black', fontsize=10, 
                   horizontalalignment='center', verticalalignment='top')
        
        # Get vector scaling
        vector_scale = self.params.get('vector_scale', 1.0)
        print(f"Vector scale: {vector_scale}")
        
        # Plot vectors at each time point
        for time in times:
            vectors = vector_data[time]
            
            # Plot each type of vector if available
            if 'u_ref' in vectors and vectors['u_ref'] is not None:
                vector = vectors['u_ref'] * vector_scale
                if np.linalg.norm(vector) > 0.001:
                    ax.quiver(time, 0, 0, vector[2], vector[0], vector[1],
                             color=COLORS["u_ref"], linewidth=2, 
                             arrow_length_ratio=0.15, alpha=0.8)
            
            if 'u_safe' in vectors and vectors['u_safe'] is not None:
                vector = vectors['u_safe'] * vector_scale
                if np.linalg.norm(vector) > 0.001:
                    ax.quiver(time, 0, 0, vector[2], vector[0], vector[1],
                             color=COLORS["u_safe"], linewidth=2, 
                             arrow_length_ratio=0.15, alpha=0.8)
            
            if 'u_filtered' in vectors and vectors['u_filtered'] is not None:
                vector = vectors['u_filtered'] * vector_scale
                if np.linalg.norm(vector) > 0.001:
                    ax.quiver(time, 0, 0, vector[2], vector[0], vector[1],
                             color=COLORS["u_filtered"], linewidth=2, 
                             arrow_length_ratio=0.15, alpha=0.8)
            
            if 'u_actual' in vectors and vectors['u_actual'] is not None:
                vector = vectors['u_actual'] * vector_scale
                if np.linalg.norm(vector) > 0.001:
                    ax.quiver(time, 0, 0, vector[2], vector[0], vector[1],
                             color=COLORS["u_actual"], linewidth=2, 
                             arrow_length_ratio=0.15, alpha=0.8)
        
        # Add axis direction indicators (unit vectors)
        axis_length = self.params.get('axis_indicator_length', 0.2) * vector_scale
        axis_offset = 1.8 * axis_length
        
        # X-axis indicator (pointing in positive Y direction)
        ax.quiver(time_start - axis_offset, 0, 0, 0, axis_length, 0,
                 color='gray', linewidth=2, arrow_length_ratio=0.2, alpha=0.7)
        ax.text(time_start - axis_offset, axis_length + 0.05, 0, 'X', 
               color='gray', fontsize=12, fontweight='bold',
               horizontalalignment='center')
        
        # Y-axis indicator (pointing in positive Z direction)  
        ax.quiver(time_start - axis_offset, 0, 0, 0, 0, axis_length,
                 color='gray', linewidth=2, arrow_length_ratio=0.2, alpha=0.7)
        ax.text(time_start - axis_offset, 0, axis_length + 0.05, 'Y', 
               color='gray', fontsize=12, fontweight='bold',
               horizontalalignment='center')
        
        # Z-axis indicator (pointing in positive X direction)
        ax.quiver(time_start - axis_offset, 0, 0, axis_length, 0, 0,
                 color='gray', linewidth=2, arrow_length_ratio=0.2, alpha=0.7)
        ax.text(time_start - axis_offset + axis_length + 0.05, 0, 0, 'Z', 
               color='gray', fontsize=12, fontweight='bold',
               horizontalalignment='center')
        
        # Create legend
        legend_elements = []
        
        # Check which vectors we actually have data for
        has_vectors = {'u_ref': False, 'u_safe': False, 'u_filtered': False, 'u_actual': False}
        for vectors in vector_data.values():
            for vec_type in has_vectors.keys():
                if vec_type in vectors and vectors[vec_type] is not None:
                    has_vectors[vec_type] = True
        
        if has_vectors['u_ref']:
            legend_elements.append(Line2D([0], [0], color=COLORS["u_ref"], lw=2, label='Reference'))
        if has_vectors['u_safe']:
            legend_elements.append(Line2D([0], [0], color=COLORS["u_safe"], lw=2, label='Safe'))
        if has_vectors['u_filtered']:
            legend_elements.append(Line2D([0], [0], color=COLORS["u_filtered"], lw=2, label='Filtered'))
        if has_vectors['u_actual']:
            legend_elements.append(Line2D([0], [0], color=COLORS["u_actual"], lw=2, label='Actual'))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        # Set axis limits
        y_lim = self.params.get('y_axis_limit', 0.5)
        z_lim = self.params.get('z_axis_limit', 0.5)
        
        ax.set_xlim(time_start - axis_offset - 0.1, time_end + 0.1)
        ax.set_ylim(-y_lim, y_lim)
        ax.set_zlim(-z_lim, z_lim)
        
        # Remove all axis labels and ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        ax.set_title(f"Control Vectors Over Time (Scale: {vector_scale:.1f})", fontsize=16)
        
        # Set view angle
        view_params = self.params.get('view_angle', {'elev': 20, 'azim': -60})
        ax.view_init(elev=view_params['elev'], azim=view_params['azim'])
        
        plt.tight_layout()
        
        # Save plot
        filename = f"time_series_vectors_{self.params.get('filename_prefix', '')}.pdf"
        plt.savefig(output_dir / filename, bbox_inches='tight', dpi=300)
        plt.close(fig)

    def process(self, data, input_mappings, output_dir):
        super().process(data, input_mappings, output_dir)
        
        # Get sample interval
        sample_interval = self.params.get('sample_interval', 0.2)  # seconds
        
        # Get topics
        u_ref_topic = input_mappings.get('u_ref')
        u_safe_topic = input_mappings.get('u_safe')
        u_filtered_topic = input_mappings.get('u_filtered')
        u_actual_topic = input_mappings.get('u_actual')
        
        # We need at least one topic to work with
        available_topics = {k: v for k, v in {
            'u_ref': u_ref_topic,
            'u_safe': u_safe_topic, 
            'u_filtered': u_filtered_topic,
            'u_actual': u_actual_topic
        }.items() if v and v in data}
        
        # Use the first available topic to determine time range
        reference_topic = list(available_topics.values())[0]
        ref_timestamps = data[reference_topic]['timestamps']
        ref_bag_start = data[reference_topic]['bag_start']
        
        # Convert timestamps to seconds correctly using the existing function
        start_time_rel = convert_bag_time_in_nanoseconds_to_seconds(ref_bag_start, ref_timestamps[0] / 1e9)
        end_time_rel = convert_bag_time_in_nanoseconds_to_seconds(ref_bag_start, ref_timestamps[-1] / 1e9)
        
        print(f"Time range: {start_time_rel:.2f}s to {end_time_rel:.2f}s")
        
        # Generate sample times
        sample_times = np.arange(start_time_rel, end_time_rel + sample_interval, sample_interval)
        
        # Extract vector data at sample times
        vector_data = {}
        max_time_diff_ns = int(sample_interval * 0.5 * 1e9)  # Allow half the sample interval as max difference
        
        for sample_time in sample_times:
            # Convert sample time back to nanoseconds using the existing function
            sample_time_ns = convert_seconds_to_bag_time_in_nanoseconds(ref_bag_start, sample_time)
            vectors = {}
            
            # Extract each type of vector if available
            for vec_name, topic in available_topics.items():
                timestamps = data[topic]['timestamps']
                messages = data[topic]['continuous']
                
                idx = self._find_closest_message_index(timestamps, sample_time_ns, max_time_diff_ns)
                if idx is not None:
                    vectors[vec_name] = self._extract_control_vector(messages[idx])
                else:
                    vectors[vec_name] = None
            
            # Only add if we have at least one vector
            if any(v is not None for v in vectors.values()):
                vector_data[sample_time] = vectors
        
        if vector_data:
            self._create_time_series_plot(vector_data, output_dir)
        else:
            print("No vector data found for time series visualization")

class ProcessorFactory:
    @staticmethod
    def create(processor_type: str, params: Dict) -> Processor:
        processors = {
            "image": ImageProcessor,
            "min_cbf_plot": MinCBFPlotProcessor,
            "u_safety_error_plot": USafetyErrorPlotProcessor,
            "u_ref_error_plot": URefErrorPlotProcessor,
            "u_sizes_plot": USizesPlotProcessor,
            "velocity_size_plot": VelocitySizePlotProcessor,
            "snapshot_visualization": SnapshotVisualizationProcessor,
            "cbf_value_image": CBFValueImageProcessor,
            "time_series_vector_visualization": TimeSeriesVectorVisualizationProcessor,
        }
        return processors[processor_type](params)


def main(config_path: Path):
    config = BagProcessingConfig(config_path)
    config.validate()
    
    extractor = DataExtractor(config)
    data = extractor.extract()
    
    processors = []
    for name, proc_cfg in config.processors.items():
        if proc_cfg.enabled:
            print(f"Creating processor: {name}")
            processor = ProcessorFactory.create(proc_cfg.type, proc_cfg.params)
            processors.append((processor, proc_cfg.input_mappings))
    
    for processor, mappings in processors:
        processor.process(data, mappings, config.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)