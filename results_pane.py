import argparse
from pathlib import Path
import os
import cv2
import numpy as np

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

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
        
        # Process sceneflow message
        #process_sceneflow(reader, target_timestamp, output_path)
        
        # Any other processing
        #additional_processing(reader, target_timestamp, output_path)
    
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
    # Extract topic name for the filename
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

    #print(image_data.shape)
    
    # Decode image using OpenCV
    #image = cv2.imdecode(image_data, CV_CONVERSION_CODES[image_msg.encoding])

    #print(CV_CONVERSION_CODES[image_msg.encoding])
    #print(image.shape)

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
    Process sceneflow message at the specified timestamp.
    
    Args:
        reader (AnyReader): Rosbag reader
        timestamp (float): Target timestamp
        output_path (Path): Output directory
    """
    # Define the sceneflow topic
    sceneflow_topic = '/sceneflow'
    
    # Find connection for the sceneflow topic
    connections = [x for x in reader.connections if x.topic == sceneflow_topic]
    
    if not connections:
        print(f"No sceneflow topic found matching '{sceneflow_topic}'")
        return
    
    # Find message closest to the timestamp
    for conn, msg_timestamp, rawdata in reader.messages(connections=connections):
        if abs(msg_timestamp - timestamp) < 0.1:  # Example threshold
            # Deserialize message
            msg = reader.deserialize(rawdata, conn.msgtype)
            
            # Process sceneflow message
            # TODO: Implement your custom processing here
            
            # Save results
            with open(output_path / 'sceneflow_results.txt', 'w') as f:
                f.write(f"Sceneflow timestamp: {msg_timestamp}\n")
                # Add more details from your processed data
            
            print(f"Processed sceneflow data saved to {output_path / 'sceneflow_results.txt'}")
            break

def additional_processing(reader, timestamp, output_path):
    """
    Perform any additional processing at the specified timestamp.
    
    Args:
        reader (AnyReader): Rosbag reader
        timestamp (float): Target timestamp
        output_path (Path): Output directory
    """
    # TODO: Add any other processing you need
    pass

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