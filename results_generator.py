import argparse
import yaml
import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Dict, List, Any, Type
from dataclasses import dataclass
from abc import ABC, abstractmethod
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

@dataclass
class TopicConfig:
    name: str
    processor: str
    params: Dict[str, Any]

@dataclass
class ProcessorConfig:
    enabled: bool
    params: Dict[str, Any]

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
        
        self.topics = [
            TopicConfig(
                name=t['name'],
                processor=t['processor'],
                params=t.get('params', {})
            ) for t in config['topics']
        ]
        
        self.processors = {
            name: ProcessorConfig(
                enabled=config['processors'][name].get('enabled', True),
                params=config['processors'][name].get('params', {})
            ) for name in config['processors']
        }
    
    def validate(self):
        if not self.bag_path.exists():
            raise ValueError(f"Bag file {self.bag_path} not found")
        self.output_dir.mkdir(parents=True, exist_ok=True)

class DataExtractor:
    def __init__(self, config: BagProcessingConfig):
        self.config = config
        self.typestore = get_typestore(Stores.ROS1_NOETIC if 
            config.ros_version == "ROS1" else Stores.LATEST)

    def extract(self) -> Dict[float, Dict[str, Any]]:
        extracted = {}
        with AnyReader([self.config.bag_path]) as reader:
            bag_start = reader.start_time
            
            for ts in self.config.timestamps:
                target_ns = bag_start + int(ts * 1e9)
                extracted[ts] = self._extract_at_time(reader, target_ns)
                
        return extracted

    def _extract_at_time(self, reader, target_time: int) -> Dict[str, Any]:
        results = {}
        for topic_cfg in self.config.topics:
            conns = [c for c in reader.connections if c.topic == topic_cfg.name]
            if not conns:
                continue
                
            closest = None
            min_diff = self.config.max_time_diff_ns
            
            for conn, msg_time, rawdata in reader.messages(connections=conns):
                current_diff = abs(msg_time - target_time)
                
                if current_diff < min_diff:
                    min_diff = current_diff
                    closest = (conn, rawdata)
                    
                if msg_time > target_time and current_diff > 2 * min_diff:
                    break
                    
            if closest:
                conn, rawdata = closest
                results[topic_cfg.name] = reader.deserialize(rawdata, conn.msgtype)
                
        return results

class Processor(ABC):
    @abstractmethod
    def process(self, timestamp: float, messages: Dict[str, Any], output_dir: Path):
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

    def process(self, timestamp: float, messages: Dict[str, Any], output_dir: Path):
        for topic, msg in messages.items():
            if not hasattr(msg, 'encoding'):
                continue
                
            filename = f"{self.params['filename_prefix']}{timestamp:.2f}.png"
            img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
            
            if msg.encoding in self.conversions and self.conversions[msg.encoding]:
                img = cv2.cvtColor(img, self.conversions[msg.encoding])
            
            cv2.imwrite(str(output_dir / filename), img)

class SceneFlowProcessor(Processor):
    def __init__(self, params: Dict):
        self.params = params
        self.vis = o3d.visualization.Visualizer()

    def process(self, timestamp: float, messages: Dict[str, Any], output_dir: Path):
        if '/scene_flow' not in messages:
            return
            
        msg = messages['/scene_flow']
        pts = np.array([[p.x, p.y, p.z] for p in msg.points])
        vecs = np.array([[v.x, v.y, v.z] for v in msg.flow_vectors]) * self.params['vector_scale']
        
        self._create_visualization(pts, vecs, output_dir / f"sceneflow_{timestamp:.2f}.png")

    def _create_visualization(self, pts, vecs, path: Path):
        self.vis.create_window()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        
        lines = []
        for i, vec in enumerate(vecs):
            lines.append([i, i+len(pts)])
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.vstack([pts, pts+vecs]))
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        self.vis.add_geometry(pcd)
        self.vis.add_geometry(line_set)
        self.vis.capture_screen_image(str(path))
        self.vis.destroy_window()

class ProcessorFactory:
    @staticmethod
    def create(processor_type: str, params: Dict) -> Processor:
        processors = {
            "image": ImageProcessor,
            "sceneflow": SceneFlowProcessor,
        }
        return processors[processor_type](params)

def main(config_path: Path):
    config = BagProcessingConfig(config_path)
    config.validate()
    
    extractor = DataExtractor(config)
    data = extractor.extract()
    
    processors = []
    for topic_cfg in config.topics:
        if config.processors[topic_cfg.processor].enabled:
            processors.append(ProcessorFactory.create(
                topic_cfg.processor,
                {**topic_cfg.params, **config.processors[topic_cfg.processor].params}
            ))
    
    for ts, messages in data.items():
        for processor in processors:
            processor.process(ts, messages, config.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)