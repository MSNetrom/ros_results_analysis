#!/usr/bin/env python3
import argparse
from pathlib import Path
from collections import defaultdict

from rosbags.highlevel import AnyReader
from rosbags.typesys import get_typestore

def list_topics(bag_path, show_details=False):
    """
    List all topics in the given ROS bag file
    
    Args:
        bag_path (Path): Path to the rosbag file
        show_details (bool): Whether to show message counts and types
    """
    bag_path = Path(bag_path)
    print(f"\nAnalyzing bag: {bag_path}")
    
    try:
        # Open the bag file without specifying type store
        with AnyReader([bag_path]) as reader:
            # Get all connections
            connections = reader.connections
            
            if not connections:
                print("No topics found in the bag file.")
                return
                
            # Group connections by topic
            topics = defaultdict(list)
            for conn in connections:
                topics[conn.topic].append(conn)
                
            # Count messages per topic
            message_counts = {}
            if show_details:
                for topic, conns in topics.items():
                    count = 0
                    for conn in conns:
                        count += sum(1 for _ in reader.messages(connections=[conn]))
                    message_counts[topic] = count
            
            # Print the results
            print(f"\nFound {len(topics)} topics in bag file:")
            print("-" * 80)
            for i, (topic, conns) in enumerate(sorted(topics.items()), 1):
                msg_types = ", ".join(sorted(set(conn.msgtype for conn in conns)))
                
                if show_details:
                    print(f"{i:3}. {topic}")
                    print(f"    Type: {msg_types}")
                    print(f"    Messages: {message_counts[topic]}")
                    print()
                else:
                    print(f"{i:3}. {topic} [{msg_types}]")
            
            print("-" * 80)
            
            # Get duration if available
            try:
                start_time = reader.start_time
                end_time = reader.end_time
                duration = end_time - start_time
                print(f"Bag duration: {duration:.2f} seconds")
            except Exception:
                # Duration might not be available for some bags
                pass
            
    except Exception as e:
        print(f"Error reading bag file: {e}")

def main():
    parser = argparse.ArgumentParser(description='List topics in a ROS bag file')
    parser.add_argument('bag_path', type=str, help='Path to the ROS bag file')
    parser.add_argument('--details', '-d', action='store_true', 
                        help='Show detailed information about each topic')
    
    args = parser.parse_args()
    list_topics(args.bag_path, args.details)

if __name__ == '__main__':
    main() 