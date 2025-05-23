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
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, v_0_values, label="$v_0$", linewidth=2)
        plt.plot(timestamps, v_1_values, label="$v_1$", linewidth=2)
        plt.legend(fontsize=14)
        plt.xlabel("Time (seconds)", fontsize=14)
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
        from mpl_toolkits.mplot3d import Axes3D

    def _extract_point_cloud(self, msg):
        """Extract XYZ points from PointCloud2 message"""
        import struct
        
        # Parse PointCloud2 data
        point_step = msg.point_step
        row_step = msg.row_step
        data = msg.data
        
        # Find XYZ field offsets
        x_offset = y_offset = z_offset = None
        for field in msg.fields:
            if field.name == 'x':
                x_offset = field.offset
            elif field.name == 'y':
                y_offset = field.offset
            elif field.name == 'z':
                z_offset = field.offset
        
        if None in (x_offset, y_offset, z_offset):
            raise ValueError("Point cloud missing XYZ fields")
        
        points = []
        for i in range(0, len(data), point_step):
            if i + 12 <= len(data):  # Ensure we have enough data for XYZ
                x = struct.unpack('f', data[i + x_offset:i + x_offset + 4])[0]
                y = struct.unpack('f', data[i + y_offset:i + y_offset + 4])[0]
                z = struct.unpack('f', data[i + z_offset:i + z_offset + 4])[0]
                
                # Filter out invalid points
                if not (np.isnan(x) or np.isnan(y) or np.isnan(z) or np.isinf(x) or np.isinf(y) or np.isinf(z)):
                    points.append([x, y, z])
        
        return np.array(points)

    def _extract_control_vector(self, msg):
        """Extract control vector from TwistStamped message"""
        return np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])

    def _create_3d_plot(self, point_cloud, reference_vec, control_vec, timestamp, output_dir):
        """Create and save 3D visualization"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Configure plot appearance
        ax.grid(True)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        
        # Plot point cloud
        if len(point_cloud) > 0:
            ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
                      c='lightblue', s=5, alpha=0.6, marker='o', edgecolors='none')
        
        # Plot origin (drone position)
        ax.scatter([0], [0], [0], c='red', s=100, marker='o')
        
        # Plot reference vector if available
        if reference_vec is not None and np.linalg.norm(reference_vec) > 0:
            ax.quiver(0, 0, 0, reference_vec[0], reference_vec[1], reference_vec[2],
                     color='red', linewidth=2, label='Reference', arrow_length_ratio=0.15)
        
        # Plot control vector if available
        if control_vec is not None and np.linalg.norm(control_vec) > 0:
            ax.quiver(0, 0, 0, control_vec[0], control_vec[1], control_vec[2],
                     color='limegreen', linewidth=2, label='Control', arrow_length_ratio=0.15)
        
        # Set axis limits based on parameters
        limits = self.params.get('axis_limits', {'x': [1, 2.5], 'y': [-1, 1], 'z': [-1, 0.5]})
        ax.set_xlim(limits['x'])
        ax.set_ylim(limits['y'])
        ax.set_zlim(limits['z'])
        
        # Configure appearance
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f"Time: {timestamp:.2f}s", fontsize=14)
        
        # Set view angle
        view_params = self.params.get('view_angle', {'elev': 30, 'azim': 170})
        ax.view_init(elev=view_params['elev'], azim=view_params['azim'])
        
        # Add legend if vectors are present
        if (reference_vec is not None and np.linalg.norm(reference_vec) > 0) or \
           (control_vec is not None and np.linalg.norm(control_vec) > 0):
            ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        # Save plot
        snapshots_dir = output_dir / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        filename = f"control_viz_t{timestamp:.2f}".replace('.', '_') + ".pdf"
        #plt.show()
        plt.savefig(snapshots_dir / filename, bbox_inches='tight')
        plt.close(fig)

    def process(self, data, input_mappings, output_dir):
        super().process(data, input_mappings, output_dir)

        print(f"Processing snapshot visualization")
        
        # Get required topics
        pc_topic = input_mappings.get('point_cloud')
        ref_topic = input_mappings.get('reference')
        ctrl_topic = input_mappings.get('control')
        
        if not pc_topic or pc_topic not in data:
            print(f"Warning: Point cloud topic {pc_topic} not found")
            return
        
        # Process each highlight timestamp
        pc_highlights = data[pc_topic]['highlights']
        pc_messages = data[pc_topic]['continuous']
        
        for idx in pc_highlights:

            print(f"Processing point cloud at idx={idx}")

            if idx >= len(pc_messages):
                continue
                
            pc_msg = pc_messages[idx]
            timestamp = convert_bag_time_in_nanoseconds_to_seconds(
                data[pc_topic]['bag_start'], 
                pc_msg.header.stamp.sec + pc_msg.header.stamp.nanosec / 1e9
            )

            print(f"Processing point cloud at t={timestamp:.2f}")
            
            # Extract point cloud
            try:
                point_cloud = self._extract_point_cloud(pc_msg)
            except Exception as e:
                print(f"Error extracting point cloud at t={timestamp:.2f}: {e}")
                continue
            
            # Extract reference vector if available
            reference_vec = None
            if ref_topic and ref_topic in data:
                ref_highlights = data[ref_topic]['highlights']
                if idx < len(ref_highlights) and ref_highlights[idx] < len(data[ref_topic]['continuous']):
                    ref_msg = data[ref_topic]['continuous'][ref_highlights[idx]]
                    reference_vec = self._extract_control_vector(ref_msg)
            
            # Extract control vector if available
            control_vec = None
            if ctrl_topic and ctrl_topic in data:
                ctrl_highlights = data[ctrl_topic]['highlights']
                if idx < len(ctrl_highlights) and ctrl_highlights[idx] < len(data[ctrl_topic]['continuous']):
                    ctrl_msg = data[ctrl_topic]['continuous'][ctrl_highlights[idx]]
                    control_vec = self._extract_control_vector(ctrl_msg)
            
            # Create 3D visualization
            self._create_3d_plot(point_cloud, reference_vec, control_vec, timestamp, output_dir)

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
        }
        return processors[processor_type](params)


def main(config_path: Path):
    config = BagProcessingConfig(config_path)
    config.validate()
    
    extractor = DataExtractor(config)
    data = extractor.extract()
    
    processors = []
    for _, proc_cfg in config.processors.items():
        if proc_cfg.enabled:
            processor = ProcessorFactory.create(proc_cfg.type, proc_cfg.params)
            processors.append((processor, proc_cfg.input_mappings))
    
    for processor, mappings in processors:
        processor.process(data, mappings, config.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)