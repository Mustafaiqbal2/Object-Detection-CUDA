import cv2
import numpy as np
import random

# Generate consistent colors based on class ID
def get_color(class_id):
    # Create a consistent color mapping based on class_id
    random.seed(class_id * 123)  # Make colors consistent for same class
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def draw_tracked_objects(frame, trackers, class_names, show_trajectories=False, show_labels=True):
    """
    Draw bounding boxes, class labels, and tracking IDs on frame
    
    Args:
        frame: The image to draw on
        trackers: Dictionary of trackers from ObjectTracker
        class_names: Dictionary or list of class names
        show_trajectories: Whether to show trajectory lines
        show_labels: Whether to show class labels
        
    Returns:
        Frame with visualizations
    """
    # Create a copy of the frame to avoid modifying the original
    result = frame.copy()
    
    for tracker_id, tracker_info in trackers.items():
        # Get box coordinates and class label
        box = tracker_info['box']
        label_id = tracker_info['label']
        
        # Skip if box is not valid
        if len(box) != 4:
            continue
        
        # Get class name
        if isinstance(class_names, dict):
            class_name = class_names.get(int(label_id), f"Class {label_id}")
        else:
            # For list-based class names
            class_name = class_names[int(label_id)] if int(label_id) < len(class_names) else f"Class {label_id}"
        
        # Get color based on class
        color = get_color(int(label_id))
        
        # Draw box
        x1, y1, x2, y2 = box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # Draw ID and class name if enabled
        if show_labels:
            label_text = f"{class_name} #{tracker_id}"
            
            # Create background for text
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(
                result, 
                (x1, y1 - text_height - 8), 
                (x1 + text_width + 5, y1), 
                color, 
                -1
            )
            # Draw text in white for better contrast
            cv2.putText(
                result, label_text, (x1 + 2, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )
        
        # Draw trajectory if enabled
        if show_trajectories:
            trajectory = tracker_info.get('trajectory', [])
            if len(trajectory) > 1:
                # Convert trajectory to points
                points = np.array([[int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)] 
                                for box in trajectory[-10:]])  # Show last 10 points
                
                # Draw trajectory line
                for i in range(1, len(points)):
                    cv2.line(result, tuple(points[i-1]), tuple(points[i]), color, 1)
    
    return result