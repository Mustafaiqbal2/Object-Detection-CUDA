import cv2
import numpy as np
# Fix: Import torch at the top level
import torch
import torchvision

class ObjectTracker:
    def __init__(self, max_disappeared=10, iou_threshold=0.3):
        """
        Initialize multi-object tracker
        
        Args:
            max_disappeared: Maximum frames before object is considered gone
            iou_threshold: IoU threshold for matching
        """
        # Initialize multi-object tracker
        self.trackers = {}
        self.next_id = 0
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        
    def update(self, boxes, labels, frame):
        """
        Update tracker with new detections
        
        Args:
            boxes: List of bounding boxes [x1, y1, x2, y2]
            labels: List of class labels
            frame: Current frame
            
        Returns:
            Dictionary of tracked objects
        """
        # Update existing trackers with new detections
        if len(boxes) == 0:
            # No detections, mark all existing trackers as disappeared
            for tracker_id in list(self.trackers.keys()):
                self.trackers[tracker_id]['disappeared'] += 1
                if self.trackers[tracker_id]['disappeared'] > self.max_disappeared:
                    del self.trackers[tracker_id]
            return self.trackers
        
        # Initialize new trackers dictionary
        updated_trackers = {}
        
        # If we currently have no trackers, create them for all boxes
        if len(self.trackers) == 0:
            for i, (box, label) in enumerate(zip(boxes, labels)):
                self._create_new_tracker(box, label, frame, updated_trackers)
        else:
            # Try to match detections with existing trackers
            # Simple IoU-based matching for demonstration
            for box, label in zip(boxes, labels):
                best_iou = self.iou_threshold
                best_id = None
                
                for tracker_id, tracker_info in self.trackers.items():
                    iou = self._calculate_iou(box, tracker_info['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_id = tracker_id
                
                if best_id is not None:
                    # Update existing tracker
                    updated_trackers[best_id] = {
                        'box': box,
                        'label': label,
                        'disappeared': 0,
                        'trajectory': self.trackers[best_id]['trajectory'] + [box]
                    }
                else:
                    # Create new tracker
                    self._create_new_tracker(box, label, frame, updated_trackers)
            
            # Check for disappeared trackers
            for tracker_id, tracker_info in self.trackers.items():
                if tracker_id not in updated_trackers:
                    tracker_info['disappeared'] += 1
                    if tracker_info['disappeared'] <= self.max_disappeared:
                        updated_trackers[tracker_id] = tracker_info
        
        self.trackers = updated_trackers
        return self.trackers
    
    def _create_new_tracker(self, box, label, frame, trackers_dict):
        """
        Create a new tracker for a detected object
        
        Args:
            box: Bounding box [x1, y1, x2, y2]
            label: Class label
            frame: Current frame
            trackers_dict: Dictionary to add the new tracker to
        """
        trackers_dict[self.next_id] = {
            'box': box,
            'label': label,
            'disappeared': 0,
            'trajectory': [box]
        }
        self.next_id += 1
    
    def _calculate_iou(self, box1, box2):
        """
        Calculate intersection over union between two bounding boxes
        
        Args:
            box1, box2: Bounding boxes in format [x1, y1, x2, y2]
            
        Returns:
            IoU value between 0 and 1
        """
        # Ensure box format is [x1, y1, x2, y2]
        if len(box1) == 4 and len(box2) == 4:
            # Calculate intersection coordinates
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            # Calculate intersection area
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            intersection = w * h
            
            # Calculate union area
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = box1_area + box2_area - intersection
            
            # Return IoU
            if union > 0:
                return intersection / union
                    
        return 0.0