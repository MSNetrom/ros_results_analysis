from collections import defaultdict
import argparse
import yaml
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

@dataclass
class ProcessorConfig:
    enabled: bool
    type: str
    params: Dict[str, Any]
    input_mappings: Dict[str, str]  # Logical -> Actual topic

class BagProcessingConfig:
    def __init__(self, config_path: Path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        bp = config['bag_processing']
        self.bag_path = Path(bp['bag_path'])
        self.timestamps = bp['timestamps']
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
        if not self.bag_path.exists():
            raise ValueError(f"Bag file {self.bag_path} not found")
        self.output_dir.mkdir(parents=True, exist_ok=True)

class DataExtractor:
    def __init__(self, config: BagProcessingConfig):
        self.config = config
        self.typestore = get_typestore(Stores.ROS1_NOETIC if 
            config.ros_version == "ROS1" else Stores.LATEST)

    def extract(self) -> Dict[str, List[Any]]:
        
        extracted = defaultdict(list)
        required_topics = set()
        
        # Collect all unique topics from all processor mappings
        for proc in self.config.processors.values():
            if proc.enabled:
                required_topics.update(proc.input_mappings.values())

        with AnyReader([self.config.bag_path]) as reader:
            bag_start = reader.start_time
            target_ns = [bag_start + int(ts * 1e9) for ts in self.config.timestamps]

            for topic in required_topics:
                conns = [c for c in reader.connections if c.topic == topic]
                if not conns:
                    continue
                
                # Get all messages for this topic
                messages = []
                times = []
                for conn, timestamp, rawdata in reader.messages(connections=conns):
                    messages.append((conn, timestamp, rawdata))
                    times.append(timestamp)
                
                # Convert to numpy arrays for vectorized operations
                times_np = np.array(times)
                
                # Find closest indices for all targets at once
                indices = np.searchsorted(times_np, target_ns, side='left')
                
                # Collect candidates
                candidates = []
                for i, idx in enumerate(indices):
                    ts_target = target_ns[i]
                    candidates_for_ts = []
                    
                    # Check previous message
                    if idx > 0:
                        candidates_for_ts.append(messages[idx-1])
                    
                    # Check current message
                    if idx < len(messages):
                        candidates_for_ts.append(messages[idx])
                    
                    # Find best candidate for this timestamp
                    if candidates_for_ts:
                        best = min(candidates_for_ts, key=lambda x: abs(x[1] - ts_target))
                        candidates.append(best)
                    else:
                        candidates.append(None)
                
                # Process candidates in batch
                for i, candidate in enumerate(candidates):
                    if candidate is None:
                        extracted[topic].append(None)
                        continue
                        
                    conn, msg_time, rawdata = candidate
                    if abs(msg_time - target_ns[i]) > self.config.max_time_diff_ns:
                        extracted[topic].append(None)
                        continue
                        
                    # Deserialize and store
                    msg = reader.deserialize(rawdata, conn.msgtype)
                    extracted[topic].append(msg)

        return extracted

class Processor(ABC):
    @abstractmethod
    def process(self, 
               timestamps: List[float], 
               data: Dict[str, List[Any]],
               input_mappings: Dict[str, str],
               output_dir: Path):
        pass

class ImageProcessor(Processor):
    def __init__(self, params: Dict):
        self.params = params
        self.conversions = {
            'rgb8': cv2.COLOR_RGB2BGR,
            'rgba8': cv2.COLOR_RGBA2BGR,
            'bgr8': None,
            'bgra8': cv2.COLOR_BGRA2BGR
        }

    def process(self, 
               timestamps: List[float], 
               data: Dict[str, List[Any]],
               input_mappings: Dict[str, str],
               output_dir: Path):
        # Use processor-specific mapping
        topic_name = input_mappings.get('color_image')
        if not topic_name or topic_name not in data:
            return

        for i, (ts, msg) in enumerate(zip(timestamps, data[topic_name])):
            if msg is None:
                continue
                
            filename = f"{self.params['filename_prefix']}{ts:.2f}.png"
            img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
            
            if msg.encoding in self.conversions and self.conversions[msg.encoding]:
                img = cv2.cvtColor(img, self.conversions[msg.encoding])
            
            cv2.imwrite(str(output_dir / filename), img)

class ProcessorFactory:
    @staticmethod
    def create(config: ProcessorConfig) -> Processor:
        processors = {
            "image": ImageProcessor,
            # Add other processor types here
        }
        return processors[config.type](config.params)

def main(config_path: Path):
    config = BagProcessingConfig(config_path)
    config.validate()
    
    extractor = DataExtractor(config)
    data = extractor.extract()
    
    processors = []
    for name, proc_cfg in config.processors.items():
        if proc_cfg.enabled:
            processor = ProcessorFactory.create(proc_cfg)
            processors.append((processor, proc_cfg.input_mappings))
    
    # Pass processor-specific mappings to each processor
    for processor, mappings in processors:
        processor.process(
            config.timestamps, 
            data, 
            mappings,
            config.output_dir
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)