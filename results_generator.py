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

class ProcessorFactory:
    @staticmethod
    def create(processor_type: str, params: Dict) -> Processor:
        processors = {
            "image": ImageProcessor,
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