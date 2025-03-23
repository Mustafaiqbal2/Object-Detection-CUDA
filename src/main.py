import sys
import os
import cv2
import time
import numpy as np
import torch
import argparse

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Import with full paths
from src.utils.cuda_utils import CUDA_AVAILABLE
from src.camera.webcam import start_webcam, get_frame, release_webcam
from src.models.detector import ObjectDetector
from src.models.tracker import ObjectTracker
from src.utils.visualization import draw_tracked_objects

# Exit if CUDA is not available
if not CUDA_AVAILABLE:
    print("ERROR: This application requires CUDA. Exiting.")
    sys.exit(1)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='High-Performance Object Detection')
    parser.add_argument('--model', type=str, default='fasterrcnn', 
                        help='Model type: fasterrcnn, fasterrcnn_resnet101, maskrcnn, yolov5, yolov8')
    parser.add_argument('--resolution', type=str, default='medium',
                        help='Input resolution: tiny (256x256), small (384x384), medium (512x512), large (640x640)')
    parser.add_argument('--interval', type=int, default=3,
                        help='Detection interval: process every N frames')
    parser.add_argument('--confidence', type=float, default=0.45,
                        help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--disable_fp16', type=str, default='false',
                        help='Disable half-precision (FP16) operations')
    args = parser.parse_args()
    
    disable_fp16 = args.disable_fp16.lower() in ('true', 't', 'yes', 'y', '1')
    
    print("Starting Object Detection and Tracking...")
    print(f"Configuration: model={args.model}, resolution={args.resolution}, interval={args.interval}, confidence={args.confidence}")
    
    # Get resolution based on argument
    resolution_map = {
        'tiny': (256, 256),
        'small': (384, 384),
        'medium': (512, 512),
        'large': (640, 640)
    }
    input_resolution = resolution_map.get(args.resolution, (384, 384))
    
    # Initialize components
    try:
        # Set CUDA optimization flags
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Start webcam with smaller resolution for better performance
        cap = start_webcam(width=640, height=480)
        
        # Initialize lighter detector
        detector = ObjectDetector(
            confidence_threshold=args.confidence,  # Balance between detection quality and speed
            model_type=args.model,   # Could use "mobilenet" if implemented
            enable_enhancement=True,     # Keep enhancement for dark environments
            disable_fp16=disable_fp16
        )
        
        # Use the input resolution for detection
        detector.input_resolution = input_resolution
        
        # Clear initial GPU memory
        torch.cuda.empty_cache()
        
        # Initialize optimal tracker settings
        tracker = ObjectTracker(max_disappeared=8, iou_threshold=0.3)
        
        print("Press 'q' to quit, 'd' to adjust detection frequency, 'e' to toggle enhancement")
        
        # Performance variables
        frame_count = 0
        fps_start_time = time.time()
        fps = 0
        processing_times = []
        
        # Dynamic detection settings
        target_fps = 30
        max_detection_interval = 5
        detection_interval = args.interval  # Start with detecting every Nth frame
        frames_since_detection = 0
        
        # Display options
        show_fps = True
        show_enhancement = True
        show_trajectories = False
        
        while True:
            loop_start = time.time()
            
            # Get frame from webcam
            frame = get_frame(cap)
            frame_count += 1
            
            # Decide whether to run detection on this frame
            run_detection = False
            frames_since_detection += 1
            if frames_since_detection >= detection_interval:
                run_detection = True
                frames_since_detection = 0
            
            # Process frame with detection or tracking
            if run_detection:
                # Run detection on this frame
                boxes, labels, scores = detector.detect(frame)
                
                # Update trackers with detections
                if len(boxes) > 0:
                    trackers = tracker.update(boxes, labels, frame)
                else:
                    trackers = tracker.update([], [], frame)
            else:
                # Use tracking only (much faster)
                trackers = tracker.update([], [], frame)
            
            # Visualize results
            result_frame = draw_tracked_objects(frame.copy(), trackers, detector.classes, 
                                            show_trajectories=show_trajectories)
            
            # Calculate and display FPS
            if frame_count % 10 == 0:
                current_time = time.time()
                fps = 10 / (current_time - fps_start_time)
                fps_start_time = current_time
                
                # Adaptive detection interval based on performance
                if frame_count > 50:  # Give the system time to stabilize
                    if fps < target_fps * 0.8 and detection_interval < max_detection_interval:
                        detection_interval += 1
                        print(f"⚡ Increasing detection interval to {detection_interval} (FPS: {fps:.1f})")
                    elif fps > target_fps * 1.2 and detection_interval > 1:
                        detection_interval -= 1
                        print(f"🔍 Decreasing detection interval to {detection_interval} (FPS: {fps:.1f})")
            
            # Periodically clean GPU memory
            if frame_count % 100 == 0:
                torch.cuda.empty_cache()
            
            # Display performance metrics
            if show_fps:
                cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(result_frame, f"Objects: {len(trackers)}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Show detection interval
                cv2.putText(result_frame, f"Det Interval: {detection_interval}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Show if frame was processed with detection
                det_status = "DETECTED" if run_detection else "TRACKED"
                cv2.putText(result_frame, det_status, (result_frame.shape[1] - 120, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Show enhancement status
                if show_enhancement:
                    enh_status = "ENH: ON" if detector.enable_enhancement else "ENH: OFF"
                    cv2.putText(result_frame, enh_status, (10, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('High-Performance Tracking', result_frame)
            
            # Record processing time
            processing_time = time.time() - loop_start
            processing_times.append(processing_time)
            if len(processing_times) > 100:
                processing_times.pop(0)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                # Cycle through detection intervals
                detection_interval = (detection_interval % max_detection_interval) + 1
                print(f"Detection interval set to {detection_interval}")
            elif key == ord('e'):
                # Toggle enhancement
                detector.enable_enhancement = not detector.enable_enhancement
                print(f"Enhancement: {'ON' if detector.enable_enhancement else 'OFF'}")
            elif key == ord('t'):
                # Toggle trajectories
                show_trajectories = not show_trajectories
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Release resources and print statistics
        if 'cap' in locals():
            release_webcam(cap)
            
        if 'processing_times' in locals() and processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            print(f"\nPerformance Statistics:")
            print(f"Average processing time: {avg_time*1000:.2f}ms")
            print(f"Effective FPS: {1/avg_time:.1f}")
            print(f"Best FPS achieved: {1/min(processing_times):.1f}")
        
        print("Object Detection and Tracking stopped.")

if __name__ == "__main__":
    main()