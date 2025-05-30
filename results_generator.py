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
import scipy as sp
import os # Added for path operations in video creation
import tempfile
from scipy.spatial.transform import Rotation as R
# ------------------------------------------------------------------
#  PointCloud-helper import (ROS 2 ▸ ROS 1 ▸ fallback)
# ------------------------------------------------------------------
try:                                # ROS 2
    from sensor_msgs_py import point_cloud2 as pc2
except ImportError:
    try:                            # ROS 1
        from sensor_msgs import point_cloud2 as pc2
    except ImportError:             # no helper available
        pc2 = None

from mpl_toolkits.mplot3d.art3d import Line3DCollection     #  NEW
from matplotlib.colors import Normalize                    #  NEW

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

class VideoCreatorUtil:
    @staticmethod
    def create_video_from_timed_images(
        image_frames_with_times: List[Tuple[Path, float]], # List of (image_path, timestamp_in_seconds)
        output_video_path: Path,
        fps: float,
        duration_sec: float,
        logger=print
    ):
        """
        Creates a video from images, adjusting frame display time based on timestamps.

        Args:
            image_frames_with_times: A list of tuples, where each tuple is (Path_to_image, timestamp_in_seconds).
                                     The list should be sorted by timestamp.
            output_video_path: Path to save the output video file.
            fps: Frames per second for the output video.
            duration_sec: Duration (in seconds) to display the last frame.
            logger: Logger function (defaults to print).
        """
        if not image_frames_with_times:
            logger(f"No image frames provided for video {output_video_path}")
            return

        # Ensure sorted by timestamp, though processors should provide it sorted.
        # This sort is crucial for correct duration calculation.
        image_frames_with_times.sort(key=lambda x: x[1])

        try:
            first_frame_img = cv2.imread(str(image_frames_with_times[0][0]))
            if first_frame_img is None:
                logger(f"Error: Could not read the first image frame: {image_frames_with_times[0][0]} for video {output_video_path}")
                return
            height, width, _ = first_frame_img.shape
        except Exception as e:
            logger(f"Error reading first image dimensions from {image_frames_with_times[0][0]}: {e}")
            return
        
        print(f"Duration of video: {duration_sec} seconds")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

        # Loop all frames of video

        for frame_idx in range(int(duration_sec * fps)):

            print(f"Frame {frame_idx} of {int(duration_sec * fps)}")

            frame_time_in_s = frame_idx / fps

            # Find the closest image frame to the current frame time
            closest_frame_time = min(image_frames_with_times, key=lambda x: abs(x[1] - frame_time_in_s))

            # Get the image frame
            img_path, _ = closest_frame_time

            frame_img = cv2.imread(str(img_path))

            video_writer.write(frame_img)

        video_writer.release()

def softmin(x: np.ndarray, k: float = 1.0):
    ins = - k * x
    logsumexp_val = sp.special.logsumexp(ins)
    return - logsumexp_val / k

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

        if not topic or topic not in data or not data[topic]['continuous']:
            print(f"ImageProcessor: Topic '{topic}' not found, not specified, or has no continuous data.")
            return

        image_output_dir = output_dir / (self.params.get('filename_prefix', 'img_') + "color_images")
        image_output_dir.mkdir(parents=True, exist_ok=True)

        timed_image_frames: List[Tuple[Path, float]] = []
        
        source_messages = data[topic]['continuous']
        source_timestamps_ns = data[topic]['timestamps']

        if len(source_messages) != len(source_timestamps_ns):
            print(f"ImageProcessor: Mismatch between continuous messages ({len(source_messages)}) and timestamps ({len(source_timestamps_ns)}) for topic {topic}. Skipping video generation.")
            return

        print(f"ImageProcessor: Processing {len(source_messages)} continuous frames for video from topic {topic}.")
        source_messages_first_timestamp_ns = source_timestamps_ns[0]

        indexes = range(len(source_messages)) if self.params.get("generate_video", False) else data[topic]['highlights']

        temp_files = []

        for i in indexes:
            msg = source_messages[i]
            timestamp_ns = source_timestamps_ns[i]
            time_in_s_absolute = (timestamp_ns - source_messages_first_timestamp_ns) / 1e9

            filename = f"{self.params.get('filename_prefix', 'img_')}{time_in_s_absolute:.3f}.png" # Use more precision for unique names

            # Check if we are in the highlights
            if i in data[topic]['highlights']:
                self._save_image(msg, image_output_dir / filename)

            if self.params.get("generate_video", False):
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                temp_file_path = Path(temp_file.name)
                temp_files.append(temp_file)

                saved_path = self._save_image(msg, temp_file_path)
                if saved_path:
                    timed_image_frames.append((saved_path, time_in_s_absolute))
            if (i + 1) % 100 == 0: # Log progress
                 print(f"ImageProcessor: Saved {i+1}/{len(source_messages)} frames for video...")


        if self.params.get("generate_video", False):
            video_filename = self.params.get("video_filename", "color_image_video.mp4")
            video_fps = float(self.params.get("video_fps", 10.0)) # FPS for the output video

            output_video_path = output_dir / video_filename
            
            VideoCreatorUtil.create_video_from_timed_images(
                image_frames_with_times=timed_image_frames,
                output_video_path=output_video_path,
                fps=video_fps,
                duration_sec=timed_image_frames[-1][1] - timed_image_frames[0][1],
                logger=print
            )

        # Delete temp files
        for temp_file in temp_files:
            temp_file.close()

    def _save_image(self, msg, output_path) -> Union[Path, None]:
        
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
        
        if msg.encoding in self.conversions and self.conversions[msg.encoding]:
            img = cv2.cvtColor(img, self.conversions[msg.encoding])
        try:
            cv2.imwrite(str(output_path), img)
            return output_path
        except Exception as e:
            print(f"Error saving image {output_path}: {e}")
            return None


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

        softmin_v_1_values = []
        k = self.params.get('k', 1.0)
        
        for msg in v_0_messages:
            float_img = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            # Extract scalar value from image (modify this based on your needs)
            scalar_value = np.min(float_img)  # or np.mean(img), img[0,0], etc.
            v_0_values.append(scalar_value)
            
        
        for msg in v_1_messages:
            float_img = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            # Extract scalar value from image (modify this based on your needs)
            softmin_v_1_values.append(softmin(float_img, k))
            scalar_value = np.min(float_img)  # or np.mean(img), img[0,0], etc.
            v_1_values.append(scalar_value)

        # Make timestamps relative to start of interval
        timestamps = self.get_timestamps_of_message_list(v_0_messages, data[topic_v_0]['bag_start'])

        # Convert to numpy arrays for easier manipulation
        v_0_values = np.array(v_0_values)
        v_1_values = np.array(v_1_values)
        softmin_v_1_values = np.array(softmin_v_1_values)

        # ──────────────────────────────────────────────────────────────────────
        # 1)  Save the static full-range PDF (unchanged behaviour)
        # ──────────────────────────────────────────────────────────────────────
        plt.figure(figsize=(12, 4))
        plt.plot(timestamps, v_0_values, label="$\psi_{min}$", linewidth=2)
        plt.plot(timestamps, v_1_values, label="$h_{min}$",  linewidth=2)
        plt.plot(timestamps, softmin_v_1_values,
                 label="$\phi=softmin(h_1, \\ldots, h_n)$", linewidth=2)
        
        # Add fill for negative portions
        fill_alpha = self.params.get('negative_fill_alpha', 0.3)
        plt.fill_between(timestamps, v_0_values, 0, where=(v_0_values < 0), 
                        color='green', alpha=fill_alpha, interpolate=True)
        plt.fill_between(timestamps, v_1_values, 0, where=(v_1_values < 0), 
                        color='blue', alpha=fill_alpha, interpolate=True)
        plt.fill_between(timestamps, softmin_v_1_values, 0, where=(softmin_v_1_values < 0), 
                        color='orange', alpha=fill_alpha, interpolate=True)
        
        # Make markers for the highlights, and set the label to the alphabet
        for i, idx in enumerate(data[topic_v_0]['highlights']):
            plt.axvline(x=timestamps[idx], color='red', linewidth=1.5)
            plt.text(timestamps[idx], plt.ylim()[1]*1.01, alphabet[i], fontsize=14, ha='center', va='bottom')

        plt.legend(fontsize=14)
        plt.xlabel("Time [s]", fontsize=14)
        plt.ylabel("Value",     fontsize=14)
        plt.title(" ", fontsize=14)
        plt.xticks(fontsize=12); plt.yticks(fontsize=12); plt.grid(True)
        static_pdf = output_dir / "min_cbf_plot.pdf"
        plt.savefig(static_pdf, bbox_inches='tight')
        plt.close()

        # ──────────────────────────────────────────────────────────────────────
        # 2)  Optional video creation with vertical marker
        # ──────────────────────────────────────────────────────────────────────
        if not self.params.get("generate_video", False):
            return

        fps           = float(self.params.get("video_fps", 10.0))
        video_name    = self.params.get("video_filename", "min_cbf_plot_video.mp4")
        video_path    = output_dir / video_name

        # ── pre-create figure & constant lines (no re-plotting) ───────────────
        fig, ax = plt.subplots(figsize=(12, 4))
        l1, = ax.plot(timestamps, v_0_values, label="$\psi_{min}$", linewidth=2)
        l2, = ax.plot(timestamps, v_1_values, label="$h_{min}$",  linewidth=2)
        l3, = ax.plot(timestamps, softmin_v_1_values,
                      label="$\phi=softmin(h_1, \\ldots, h_n)$", linewidth=2)
        
        # Add fill for negative portions (same as static plot)
        ax.fill_between(timestamps, v_0_values, 0, where=(v_0_values < 0), 
                       color='green', alpha=fill_alpha, interpolate=True)
        ax.fill_between(timestamps, v_1_values, 0, where=(v_1_values < 0), 
                       color='blue', alpha=fill_alpha, interpolate=True)
        ax.fill_between(timestamps, softmin_v_1_values, 0, where=(softmin_v_1_values < 0), 
                       color='orange', alpha=fill_alpha, interpolate=True)
        
        ax.set_xlabel("Time [s]", fontsize=14)
        ax.set_ylabel("Value",     fontsize=14)
        ax.set_title("Min CBF Plot", fontsize=14)
        ax.grid(True); ax.legend(fontsize=14)
        ax.tick_params(labelsize=12)

        # we will only update the x-position of this vertical line
        marker = ax.axvline(x=timestamps[0], color='red', linewidth=1.5)

        timed_frames, tmp_files = [], []
        start_t = timestamps[0]

        for t in timestamps:
            marker.set_xdata([t, t])
            fig.canvas.draw_idle()

            # save current frame to a temporary png
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig.savefig(tmp.name, dpi=200, bbox_inches='tight')
            tmp_files.append(tmp)
            timed_frames.append((Path(tmp.name), t - start_t))

        plt.close(fig)

        # ── stitch the pngs into a video ──────────────────────────────────────
        VideoCreatorUtil.create_video_from_timed_images(
            image_frames_with_times = timed_frames,
            output_video_path       = video_path,
            fps                     = fps,
            duration_sec            = timed_frames[-1][1] - timed_frames[0][1],
            logger                  = print
        )

        # clean-up tmp files
        for f in tmp_files:
            f.close()

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
        plt.figure(figsize=(12, 4))
        plt.plot(timestamps, distances_filtered, label="Filtered", color=COLORS["u_filtered"], linewidth=2)
        plt.plot(timestamps, distances_actual, label="Final (clamped)", color=COLORS["u_actual"], linewidth=2)
        plt.legend(fontsize=14)
        plt.xlabel("Time [s]", fontsize=14)
        plt.ylabel("Distance [$m/s^2$]", fontsize=14)
        plt.title("Difference between safe and filtered/final control vector", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)

        plt.savefig(output_dir / "u_safety_error_plot.pdf", bbox_inches='tight')
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

        plt.figure(figsize=(12, 4))
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

        plt.savefig(output_dir / "u_ref_error_plot.pdf", bbox_inches='tight')
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

        plt.figure(figsize=(12, 4))
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

        plt.savefig(output_dir / "u_sizes_plot.pdf", bbox_inches='tight')
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
        plt.figure(figsize=(12, 4))
        plt.plot(timestamps, velocity_sizes, label="Velocity magnitude", linewidth=2)
        plt.legend(fontsize=14)
        plt.xlabel("Time [s]", fontsize=14)
        plt.ylabel("Velocity [m/s]", fontsize=14)
        plt.title("Robot velocity magnitude over time", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        
        plt.savefig(output_dir / "velocity_size_plot.pdf", bbox_inches='tight')
        plt.close()

class AngularVelocitySizePlotProcessor(Processor):

    def process(self, data, input_mappings, output_dir):
        super().process(data, input_mappings, output_dir)

        topic = input_mappings.get('odometry')
        
        if not topic or topic not in data:
            return
        
        odometry_messages = data[topic]['continuous']
        
        # Extract angular velocity values from odometry messages
        angular_velocity_values = []
        for msg in odometry_messages:
            angular_vel = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
            angular_velocity_values.append(angular_vel)
        
        angular_velocity_values = np.array(angular_velocity_values)
        
        # Calculate angular velocity magnitudes (sizes)
        angular_velocity_sizes = np.linalg.norm(angular_velocity_values, axis=1)
        
        # Get timestamps relative to start
        timestamps = self.get_timestamps_of_message_list(odometry_messages, data[topic]['bag_start'])
        
        # Create angular velocity magnitude plot
        plt.figure(figsize=(12, 4))
        plt.plot(timestamps, angular_velocity_sizes, label="Angular velocity magnitude", linewidth=2)
        plt.legend(fontsize=14)
        plt.xlabel("Time [s]", fontsize=14)
        plt.ylabel("Angular velocity [rad/s]", fontsize=14)
        plt.title("Robot angular velocity magnitude over time", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        
        plt.savefig(output_dir / "angular_velocity_size_plot.pdf", bbox_inches='tight')
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

    def _create_3d_plot(self, points, flow_vectors, u_ref_vec, 
                        u_safe_vec, u_filtered_vec, u_actual_vec, 
                        timestamp, output_path):
        
        """Create and save 3D visualization with sceneflow"""
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
        #snapshots_dir = output_dir / "snapshots"
        #snapshots_dir.mkdir(parents=True, exist_ok=True)
        #filename = f"{filename_prefix}sceneflow_viz_t{timestamp:.2f}".replace('.', '_') + ".pdf"
        plt.savefig(output_path, bbox_inches='tight')
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

        indexes = range(len(sceneflow_messages)) if self.params.get("generate_video", False) else sceneflow_highlights
        #timestamps = self.get_timestamps_of_message_list(sceneflow_messages, data[sceneflow_topic]['bag_start'])

        temp_files = []
        timestamp_start = convert_bag_time_in_nanoseconds_to_seconds(
            data[sceneflow_topic]['bag_start'], 
            data[sceneflow_topic]['timestamps'][0] / 1e9
        )

        # Save paths and timestamps
        timed_image_frames = []
        
        for idx in indexes:

            print(f"Sceneflow message {idx} of {len(indexes)}")
                
            sceneflow_msg = sceneflow_messages[idx]
            timestamp = convert_bag_time_in_nanoseconds_to_seconds(
                data[sceneflow_topic]['bag_start'], 
                data[sceneflow_topic]['timestamps'][idx] / 1e9
            ) - timestamp_start
            
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
            if idx in sceneflow_highlights:
                filename = f"{self.params.get('filename_prefix', 'sceneflow_viz_t')}{timestamp:.2f}.pdf"
                output_path = output_dir / filename
                self._create_3d_plot(points, flow_vectors, u_ref_vec, u_safe_vec, u_filtered_vec, u_actual_vec, 
                                 timestamp, output_path)
                
            if self.params.get("generate_video", False):
            
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                temp_file_path = Path(temp_file.name)
                temp_files.append(temp_file)
                timed_image_frames.append((temp_file_path, timestamp))

                self._create_3d_plot(points, flow_vectors, u_ref_vec, u_safe_vec, u_filtered_vec, u_actual_vec, 
                                    timestamp, temp_file_path)

        if self.params.get("generate_video", False):
            fps        = float(self.params.get("video_fps", 1.0))
            video_name = self.params.get("video_filename", "snapshot_viz_video.mp4")
            VideoCreatorUtil.create_video_from_timed_images(
                image_frames_with_times = timed_image_frames,
                output_video_path       = output_dir / video_name,
                fps                     = fps,
                duration_sec            = timed_image_frames[-1][1] - timed_image_frames[0][1],
                logger                  = print
            )

        for f in temp_files:
            f.close()


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

        indexes = range(len(v_0_data['continuous'])) if self.params.get("generate_video", False) else v_0_data['highlights']
        temp_files = []
        timed_image_frames_0 = []
        timed_image_frames_1 = []

        for idx in indexes:


            v_0_np = np.frombuffer(v_0_data['continuous'][idx].data, dtype=np.uint8).reshape(v_0_data['continuous'][idx].height, v_0_data['continuous'][idx].width, -1)
            v_1_np = np.frombuffer(v_1_data['continuous'][idx].data, dtype=np.uint8).reshape(v_1_data['continuous'][idx].height, v_1_data['continuous'][idx].width, -1)

            if idx in v_0_data['highlights']:
                cv2.imwrite(str(v_0_dir / f"v_0_image_{timestamps[idx]:.2f}.png"), v_0_np)
                cv2.imwrite(str(v_1_dir / f"v_1_image_{timestamps[idx]:.2f}.png"), v_1_np)

            if self.params.get("generate_video", False):
                temp_file_0 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                temp_file_1 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                temp_file_path_0 = Path(temp_file_0.name)
                temp_file_path_1 = Path(temp_file_1.name)

                temp_files.append(temp_file_0)
                temp_files.append(temp_file_1)
                timed_image_frames_0.append((temp_file_path_0, timestamps[idx]))
                timed_image_frames_1.append((temp_file_path_1, timestamps[idx]))

                cv2.imwrite(str(temp_file_path_0), v_0_np)
                cv2.imwrite(str(temp_file_path_1), v_1_np)
        
        if self.params.get("generate_video", False):
            fps        = float(self.params.get("video_fps", 1.0))
            video_name = self.params.get("video_filename", "snapshot_viz_video.mp4")

            video_name_0 = "0_" + video_name
            video_name_1 = "1_" + video_name

            VideoCreatorUtil.create_video_from_timed_images(
                image_frames_with_times = timed_image_frames_0,
                output_video_path       = output_dir / video_name_0,
                fps                     = fps,
                duration_sec            = timed_image_frames_0[-1][1] - timed_image_frames_0[0][1],
                logger                  = print
            )

            VideoCreatorUtil.create_video_from_timed_images(
                image_frames_with_times = timed_image_frames_1,
                output_video_path       = output_dir / video_name_1,
                fps                     = fps,
                duration_sec            = timed_image_frames_1[-1][1] - timed_image_frames_1[0][1],
                logger                  = print
            )

        for f in temp_files:
            f.close()

class TimeSeriesVectorVisualizationProcessor(Processor):

    def __init__(self, params: Dict):
        super().__init__(params)

    def _find_closest_message_index(self,
                                    timestamps,
                                    target_time_ns,
                                    max_diff_ns):
        """Return index of the message closest to target_time_ns (or None)."""
        if not timestamps:                         # empty list
            return None

        idx = bisect.bisect_left(timestamps, target_time_ns)
        candidates = []

        if idx > 0:
            candidates.append((idx - 1,
                               abs(timestamps[idx - 1] - target_time_ns)))
        if idx < len(timestamps):
            candidates.append((idx,
                               abs(timestamps[idx]     - target_time_ns)))  # <- fixed

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
            legend_height = self.params.get('legend_height', 0.65)
            legend_width = self.params.get('legend_width', 0.8)
            ax.legend(handles=legend_elements, loc='upper right', 
                      bbox_to_anchor=(legend_width, legend_height),
                      fontsize=12)
        
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

class PointCloudPathVisualizationProcessor(Processor):
    """
    Build one aggregated point-cloud in the fixed frame and draw the
    vehicle trajectory through it.

    params supported in YAML
    -------------------------------------------------------------------
    sample_every_n       : use every n-th PointCloud2                   (10)
    max_time_diff        : s, tolerance cloud ↔ odom / u_*              (0.05)
    max_points_per_cloud : random down-sample each cloud               (5000)
    distance_threshold   : m, keep only points within radius (None=off)
    point_size           : scatter size of cloud points                   (8)
    point_color_by       : 'time' | 'distance' | 'constant'          ('constant')
    point_cmap           : matplotlib cmap for points               ('viridis')
    path_linewidth       : width of trajectory line                      (3.0)
    cmap                 : colormap for colouring CBF correction    ('viridis')
    axis_limits          : dict, e.g. {'x':[-5,5], 'y':[-5,5], 'z':[-2,2]}
    view_angle           : dict, e.g. {'elev':20, 'azim':-60}
    """
    def __init__(self, params: Dict):
        super().__init__(params)
        if pc2 is None:
            raise ImportError(
                "Package `sensor_msgs_py` required. "
                "Install with `pip install sensor_msgs_py`."
            )

    # ---------- helpers -------------------------------------------------
    def _find_closest_msg(self,
                          stamps: List[int],            # nanoseconds
                          target: int,
                          max_diff_ns: int) -> Union[int, None]:
        """index of msg with time closest to target (None if too far)"""
        idx = bisect.bisect_left(stamps, target)
        cand = []
        if idx > 0:
            cand.append((idx-1, abs(stamps[idx-1] - target)))
        if idx < len(stamps):
            cand.append((idx,   abs(stamps[idx]   - target)))
        if not cand:
            return None
        best, diff = min(cand, key=lambda x: x[1])
        return best if diff <= max_diff_ns else None

    def _pose_from_odom(self, odom_msg):
        p = np.array([odom_msg.pose.pose.position.x,
                      odom_msg.pose.pose.position.y,
                      odom_msg.pose.pose.position.z])
        q = np.array([odom_msg.pose.pose.orientation.x,
                      odom_msg.pose.pose.orientation.y,
                      odom_msg.pose.pose.orientation.z,
                      odom_msg.pose.pose.orientation.w])
        R_bl_in_odom = R.from_quat(q)          # base_link  →  odom
        return p, R_bl_in_odom

    def _xyz_from_cloud(self, pc_msg) -> np.ndarray:
        """
        Return an (N,3) array with xyz coordinates of valid points.
        Works with:
          • sensor_msgs[_py].point_cloud2 helper (ROS-runtime messages)
          • pure-NumPy fallback (messages read with rosbags)
        """
        # --- try helper first ------------------------------------------
        if pc2 is not None:
            try:
                return np.asarray(list(
                    pc2.read_points(pc_msg,
                                    field_names=('x', 'y', 'z'),
                                    skip_nans=True)),
                    dtype=np.float32)
            except (AssertionError, AttributeError, TypeError):
                # not a roslib message → silently drop to fallback
                pass

        # ---------- pure-NumPy fallback --------------------------------
        # Assumes that x, y, z are the first three float32 fields,
        # which is true for Velodyne, Realsense, Ouster, …
        if pc_msg.point_step % 4:          # not multiple of float32 bytes
            raise ValueError("Unsupported PointCloud2 layout "
                             "(cannot decode without helper library).")

        n_pts   = len(pc_msg.data) // pc_msg.point_step
        floats  = np.frombuffer(pc_msg.data, dtype=np.float32)

        # correct endianness
        if pc_msg.is_bigendian:
            floats = floats.byteswap()

        floats  = floats.reshape(n_pts, pc_msg.point_step // 4)
        xyz     = floats[:, :3]            # first three floats → x,y,z

        # remove NaN / Inf rows
        mask = np.isfinite(xyz).all(axis=1)
        return xyz[mask]

    # ---------- processing -------------------------------------------------
    def process(self,
                data: Dict[str, Any],
                input_mappings: Dict[str, str],
                output_dir: Path):

        super().process(data, input_mappings, output_dir)

        pc_topic   = input_mappings.get('point_cloud')
        odo_topic  = input_mappings.get('odometry')
        u_ref_topic  = input_mappings.get('u_ref')     #  NEW (optional)
        u_safe_topic = input_mappings.get('u_safe')    #  NEW (optional)

        if pc_topic not in data or odo_topic not in data:
            print("[PointCloudPath] topic missing – nothing done.")
            return

        pc_msgs      = data[pc_topic]['continuous']
        pc_stamps    = data[pc_topic]['timestamps']          # ns
        odom_msgs    = data[odo_topic]['continuous']
        odom_stamps  = data[odo_topic]['timestamps']

        # optional control topics
        have_cbf = u_ref_topic in data and u_safe_topic in data
        if have_cbf:
            u_ref_msgs   = data[u_ref_topic]['continuous']
            u_ref_stamps = data[u_ref_topic]['timestamps']
            u_safe_msgs  = data[u_safe_topic]['continuous']
            u_safe_stamps= data[u_safe_topic]['timestamps']

        # --------------- parameters ----------------------------------
        every_n        = int(self.params.get('sample_every_n',          10))
        max_dt_ns      = int(self.params.get('max_time_diff', 0.05) * 1e9)
        max_pts_cloud  = int(self.params.get('max_points_per_cloud', 5000))
        dist_thresh    = self.params.get('distance_threshold',       None)

        # visual appearance
        point_size     = int(self.params.get('point_size',               8))
        point_color_by = str(self.params.get('point_color_by',
                                         'constant')).strip().lower()
        point_cmap     =      self.params.get('point_cmap',       'viridis')
        path_lw        = float(self.params.get('path_linewidth',        3.0))
        cmap_name      =      self.params.get('cmap',             'viridis')

        # --------------- aggregate point-cloud -----------------------
        agg_points, agg_colours = [], []          # arrays collected here
        path_pos_cloud = []
        for i in range(0, len(pc_msgs), every_n):
            pc_msg, pc_stamp = pc_msgs[i], pc_stamps[i]

            # ---- match odometry pose --------------------------------
            j = self._find_closest_msg(odom_stamps, pc_stamp, max_dt_ns)
            if j is None:
                continue


            p_odom, R_odom = self._pose_from_odom(odom_msgs[j])
            path_pos_cloud.append(p_odom)

            # ---- read xyz -------------------------------------------
            xyz = self._xyz_from_cloud(pc_msg)
            if xyz.size == 0:
                continue


            if xyz.shape[0] > max_pts_cloud:
                xyz = xyz[np.random.choice(xyz.shape[0],
                                           max_pts_cloud, replace=False)]
                
            xyz_odom = R_odom.apply(xyz) + p_odom          # → odom frame

            # ---- distance filter (optional) -------------------------
            if dist_thresh is not None:
                mask = np.linalg.norm(xyz_odom, axis=1) <= dist_thresh
                xyz_odom = xyz_odom[mask]
            if not xyz_odom.size:
                continue

            # ---- colour per point -----------------------------------
            if point_color_by == 'time':
                # Convert to elapsed seconds from first timestamp
                t_seconds = (pc_stamp - pc_stamps[0]) / 1e9
                agg_colours.append(np.full(xyz_odom.shape[0], t_seconds, dtype=float))
            elif point_color_by == 'distance':
                agg_colours.append(np.linalg.norm(xyz_odom, axis=1))
            else:                                      # 'constant'
                agg_colours.append(np.zeros(xyz_odom.shape[0]))

            agg_points.append(xyz_odom)

        if len(agg_points) == 0:                   # <=  more robust check
            print("[PointCloudPath] no valid data collected.")
            return

        agg_points  = np.concatenate(agg_points,  axis=0)
        agg_colours = np.concatenate(agg_colours, axis=0)
        path_pos_cloud = np.vstack(path_pos_cloud)

        # --------------- full path & CBF correction ------------------
        path_all   = []
        corr_vals  = []              # ‖u_safe - u_ref‖
        for k, odom_msg in enumerate(odom_msgs):
            p_odom, _ = self._pose_from_odom(odom_msg)
            path_all.append(p_odom)

            if have_cbf:
                t_stamp = odom_stamps[k]
                ir = self._find_closest_msg(u_ref_stamps,  t_stamp, max_dt_ns)
                is_ = self._find_closest_msg(u_safe_stamps, t_stamp, max_dt_ns)

                if ir is not None and is_ is not None:
                    u_ref_vec   = np.array([
                        u_ref_msgs[ir].twist.linear.x,
                        u_ref_msgs[ir].twist.linear.y,
                        u_ref_msgs[ir].twist.linear.z])
                    u_safe_vec  = np.array([
                        u_safe_msgs[is_].twist.linear.x,
                        u_safe_msgs[is_].twist.linear.y,
                        u_safe_msgs[is_].twist.linear.z])
                    corr_vals.append(np.linalg.norm(u_safe_vec - u_ref_vec))
                else:
                    corr_vals.append(np.nan)   # no data
            else:
                corr_vals.append(np.nan)

        path_all  = np.vstack(path_all)
        corr_vals = np.array(corr_vals)
        # replace NaN (no data) by 0 so that they appear in colour-scale
        corr_vals[np.isnan(corr_vals)] = 0.0

        # --------------- plot ----------------------------------------
        fig = plt.figure(figsize=(10, 8))
        ax  = fig.add_subplot(111, projection='3d')

        # transparent panes (remove grey background)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_facecolor((1.0, 1.0, 1.0, 0.0))

        # --- aggregated cloud ---------------------------------------
        points_scatter = ax.scatter(agg_points[:, 0], agg_points[:, 1], agg_points[:, 2],
                   s=point_size, c=agg_colours, cmap=point_cmap, alpha=0.9)

        # --- colourised path ---------------------------------------
        segments = np.stack([path_all[:-1], path_all[1:]], axis=1)
        norm     = Normalize(vmin=np.min(corr_vals), vmax=np.max(corr_vals))
        lc       = Line3DCollection(segments, cmap=cmap_name, norm=norm,
                                    linewidths=path_lw)        # ← uses param
        lc.set_array(corr_vals[1:])
        ax.add_collection(lc)

        # colourbar ---------------------------------------------------
        cb = fig.colorbar(lc, ax=ax, fraction=0.03, pad=-0.2)
        cb.set_label("$\|u_{safe} - u_{ref}\|$ [$m/s^2$]", fontsize=12)

        # Colorabr for point cloud
        # Point cloud colorbar (only if not constant coloring)
        point_cmap_padding = self.params.get('point_cmap_padding', -0.2)
        if point_color_by != 'constant':
            cb_points = fig.colorbar(points_scatter, ax=ax, 
                                     fraction=0.04, pad=point_cmap_padding, 
                                     location='top')
            
            if point_color_by == 'time':
                cb_points.set_label("Time [s]", fontsize=12)
            elif point_color_by == 'distance':
                cb_points.set_label("Distance from quadrotor [m]", fontsize=12)

        # optional axis limits / view
        axis_limits = self.params.get('axis_limits', None)
        if axis_limits:
            ax.set_xlim(axis_limits.get('x', ax.get_xlim()))
            ax.set_ylim(axis_limits.get('y', ax.get_ylim()))
            ax.set_zlim(axis_limits.get('z', ax.get_zlim()))
        view_angle = self.params.get('view_angle', None)
        if view_angle:
            ax.view_init(elev=view_angle.get('elev', None),
                         azim=view_angle.get('azim', None))

        #ax.set_title("Quadrotor trajectory through aggregated point-cloud",
        #             fontsize=14)
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
        ax.set_box_aspect([1, 1, 1]); ax.grid(False)
        plt.tight_layout()

        fig_path = output_dir / "pointcloud_path_viz.pdf"
        plt.savefig(fig_path)
        plt.close(fig)
        print(f"[PointCloudPath] saved {fig_path}")

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
            "angular_velocity_size_plot": AngularVelocitySizePlotProcessor,
            "pointcloud_path_visualization": PointCloudPathVisualizationProcessor,
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