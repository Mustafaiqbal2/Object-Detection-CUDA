import cv2
import numpy as np
import time
import sys
import os

# Import PyTorch
import torch
import torchvision

# Define COCO categories for fallback
COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Import CUDA utilities
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.cuda_utils import CUDA_AVAILABLE, CUDA_OPS_AVAILABLE, cuda_ops

class ObjectDetector:
    def __init__(self, confidence_threshold=0.5, model_type="fasterrcnn", enable_enhancement=True, disable_fp16=False):
        """
        Initialize object detector with a specified model type
        
        Args:
            confidence_threshold: Detection confidence threshold
            model_type: The type of model to use
            enable_enhancement: Enable or disable adaptive light enhancement
            disable_fp16: Disable half-precision (FP16) if True
        """
        print(f"Torch version in detector.py: {torch.__version__}")
        print(f"CUDA in detector.py: {torch.cuda.is_available()}")
        
        if not torch.cuda.is_available():
            raise ImportError("CUDA is required for this application")
            
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda')
        print(f"Detector using device: {self.device}")
        
        # Choose model based on specified type
        self.model_type = model_type.lower()  # Convert to lowercase for case-insensitivity
        
        try:
            if self.model_type == "fasterrcnn_resnet101":
                # Try different ResNet-101 variants based on what's available
                print("Loading Faster R-CNN with ResNet-101 backbone...")
                try:
                    # Try V2 version first (newer torchvision)
                    self.weights = torchvision.models.detection.FasterRCNN_ResNet101_FPN_V2_Weights.DEFAULT
                    self.model = torchvision.models.detection.fasterrcnn_resnet101_fpn_v2(weights=self.weights)
                except AttributeError:
                    try:
                        # Try regular version (older torchvision)
                        print("V2 not available, trying regular FPN version...")
                        self.weights = torchvision.models.detection.FasterRCNN_ResNet101_FPN_Weights.DEFAULT
                        self.model = torchvision.models.detection.fasterrcnn_resnet101_fpn(weights=self.weights)
                    except AttributeError:
                        # Fallback to non-weighted version
                        print("Using ResNet-101 without pre-defined weights...")
                        self.model = torchvision.models.detection.fasterrcnn_resnet101_fpn(pretrained=True)
                        self.classes = COCO_INSTANCE_CATEGORY_NAMES  # Define this at the top of the file
                
                self.model.to(self.device)
                self.model.eval()
                
                # Try to get class names from weights if available
                try:
                    self.classes = self.weights.meta["categories"]
                except (AttributeError, KeyError):
                    # Fallback to COCO class names
                    self.classes = ["__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", 
                               "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", 
                               "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
                               "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", 
                               "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", 
                               "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
                               "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", 
                               "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", 
                               "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", 
                               "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", 
                               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", 
                               "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
                
                print(f"Loaded Faster R-CNN ResNet-101 model with {len(self.classes)} classes")
                
            elif self.model_type == "maskrcnn":
                # Mask R-CNN for instance segmentation
                print("Loading Mask R-CNN...")
                self.weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
                self.model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=self.weights)
                self.model.to(self.device)
                self.model.eval()
                self.classes = self.weights.meta["categories"]
                print(f"Loaded Mask R-CNN model with {len(self.classes)} classes")
                
            elif self.model_type == "fasterrcnn" or self.model_type == "default":
                # Standard Faster R-CNN
                print("Loading Faster R-CNN with ResNet-50 backbone...")
                self.weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
                self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=self.weights)
                self.model.to(self.device)
                self.model.eval()
                self.classes = self.weights.meta["categories"]
                print(f"Loaded Faster R-CNN model with {len(self.classes)} classes")
                
            elif self.model_type == "yolov5":
                # YOLOv5 model (requires torch hub)
                print("Loading YOLOv5...")
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True, trust_repo=True)
                self.model.to(self.device)
                self.model.eval()
                self.classes = self.model.names
                print(f"Loaded YOLOv5 model with {len(self.classes)} classes")
                
            elif self.model_type == "yolov8":
                # YOLOv8 model (requires ultralytics package)
                try:
                    from ultralytics import YOLO
                    print("Loading YOLOv8...")
                    self.model = YOLO('yolov8l.pt')  # 'l' for large
                    self.classes = self.model.names
                    print(f"Loaded YOLOv8 model with {len(self.classes)} classes")
                except ImportError:
                    print("Ultralytics package not found. Install with: pip install ultralytics")
                    raise
                    
            else:
                # Unknown model type, fall back to Faster R-CNN
                print(f"Unknown model type '{model_type}'. Falling back to Faster R-CNN.")
                self.model_type = "fasterrcnn"
                self.weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
                self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=self.weights)
                self.model.to(self.device)
                self.model.eval()
                self.classes = self.weights.meta["categories"]
                
        except Exception as e:
            print(f"Failed to load specified model: {e}")
            print("Falling back to Faster R-CNN with ResNet-50.")
            self.model_type = "fasterrcnn"
            self.weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=self.weights)
            self.model.to(self.device)
            self.model.eval()
            self.classes = self.weights.meta["categories"]
            
        # Print CUDA status
        print(f"CUDA optimized operations available: {CUDA_OPS_AVAILABLE}")
        
        # In your detector.py, after loading the model
        # Convert model to TorchScript for better performance
        if self.model_type in ["fasterrcnn", "fasterrcnn_resnet101", "maskrcnn"]:
            try:
                # Trace the model with an example input
                dummy_input = torch.zeros((3, 640, 640), device=self.device)
                self.model = torch.jit.trace_module(
                    self.model, 
                    {"forward": [dummy_input.unsqueeze(0)]}
                )
                print("Model optimized with TorchScript")
            except Exception as e:
                print(f"Could not optimize model with TorchScript: {e}")
        
        # Set FP16 flag based on argument and environment
        self.use_fp16 = False
        if os.environ.get('DISABLE_FP16') or disable_fp16:
            self.use_fp16 = False
            print("Half-precision (FP16) disabled by configuration")
        else:
            from src.utils.cuda_utils import FP16_SUPPORTED
            self.use_fp16 = FP16_SUPPORTED
        
        # Attempt to convert model to half precision only if enabled
        if self.use_fp16:
            try:
                self.model = self.model.half()
                print("Model converted to half precision (FP16)")
            except Exception as e:
                print(f"Could not convert model to half precision: {e}")
                self.model = self.model.float()
                self.use_fp16 = False
        else:
            # Ensure model is in full precision
            self.model = self.model.float()
            print("Using full precision (FP32)")

        # Warm up the model with a dummy input
        self._warmup()
        
        # Add enhancement control
        self.enable_enhancement = enable_enhancement
        self.last_detection_count = 0
        self.enhancement_history = []  # Track enhancement effectiveness
        
    def _warmup(self):
        """Warm up the model with a dummy input"""
        print("Warming up model...")
        try:
            # Create dummy input with correct precision
            dummy_input = torch.zeros(1, 3, 384, 384, device=self.device)
            if hasattr(self, 'use_fp16') and self.use_fp16:
                dummy_input = dummy_input.half()
            
            # Run inference
            with torch.no_grad():
                for _ in range(2):  # Run twice for better warmup
                    self.model([dummy_input])
            
            print("Model warmup complete")
        except Exception as e:
            print(f"Warmup failed: {e}")
    
    def detect(self, frame):
        """Detect objects with adaptive light optimization"""
        start_time = time.time()
        
        # Store original frame
        original_frame = frame.copy()
        enhanced_frame = None
        
        # Analyze lighting conditions
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = cv2.mean(gray)[0]
        brightness_std = cv2.meanStdDev(gray)[1][0][0]
        
        # Determine if enhancement should be applied
        apply_enhancement = (
            self.enable_enhancement and 
            (mean_brightness < 80 or  # Too dark
            (mean_brightness < 120 and brightness_std < 40))  # Low contrast
        )
        
        # Apply enhancement if needed
        if apply_enhancement:
            # Create enhanced version
            enhanced_frame = self._enhance_image(frame)
            
            # Use the enhanced frame for detection
            frame_to_process = enhanced_frame
        else:
            # Use original frame
            frame_to_process = original_frame
        
        # Perform object detection
        boxes, labels, scores = self._detect_core(frame_to_process)
        
        # If enhancement yielded poor results, try with original
        if apply_enhancement and len(boxes) < self.last_detection_count and len(self.enhancement_history) > 5:
            # If enhancement has been making things worse
            if sum(self.enhancement_history[-5:]) < 0:
                print("Enhancement seems ineffective, trying original image")
                boxes_orig, labels_orig, scores_orig = self._detect_core(original_frame)
                
                # If original is better, use those results
                if len(boxes_orig) > len(boxes):
                    print(f"Original image better: {len(boxes_orig)} vs {len(boxes)} detections")
                    boxes, labels, scores = boxes_orig, labels_orig, scores_orig
                    self.enhancement_history.append(-1)  # Mark as negative
                else:
                    self.enhancement_history.append(1)  # Mark as positive
            else:
                self.enhancement_history.append(1 if len(boxes) > 0 else -1)
        else:
            self.enhancement_history.append(1 if len(boxes) > 0 else -1)
            
        # Keep enhancement history manageable
        if len(self.enhancement_history) > 20:
            self.enhancement_history = self.enhancement_history[-20:]
        
        # Update detection count
        self.last_detection_count = len(boxes)
        
        detection_time = (time.time() - start_time) * 1000
        print(f"Detection time: {detection_time:.1f}ms, Objects: {len(boxes)}, "
              f"Enhanced: {apply_enhancement}")
        
        return boxes, labels, scores
    
    def _enhance_image(self, frame):
        """Apply image enhancement for better object detection"""
        try:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            
            # Split the image into channels and enhance each
            channels = cv2.split(frame)
            enhanced_channels = []
            
            for channel in channels:
                enhanced_channels.append(clahe.apply(channel))
                
            # Merge the enhanced channels
            enhanced = cv2.merge(enhanced_channels)
            
            # Increase contrast
            alpha = 1.3  # Contrast control (1.0 means no change)
            beta = 10    # Brightness control
            enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
            
            return enhanced
        except Exception as e:
            print(f"Enhancement failed: {e}")
            return frame
    
    def _detect_core(self, frame):
        """Core object detection logic"""
        start_time = time.time()
        
        # Resize the frame to improve performance
        h, w = frame.shape[:2]
        input_size = getattr(self, 'input_resolution', (384, 384))
        
        if self.model_type in ["fasterrcnn", "fasterrcnn_resnet101", "maskrcnn"]:
            # Convert BGR to RGB (PyTorch models expect RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize for faster processing
            frame_resized = cv2.resize(frame_rgb, input_size)
            
            # Convert to tensor with proper normalization
            frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
            frame_tensor = frame_tensor.to(self.device)
            
            # Use correct precision for input tensor
            if hasattr(self, 'use_fp16') and self.use_fp16:
                frame_tensor = frame_tensor.half()
            
            # Forward pass with CUDA optimization
            with torch.no_grad():
                predictions = self.model([frame_tensor])[0]
            
            # Extract predictions
            boxes = predictions['boxes'].cpu().numpy()
            scores = predictions['scores'].cpu().numpy()
            labels = predictions['labels'].cpu().numpy()
            
            # Filter by confidence
            mask = scores >= self.confidence_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]
            
            # Scale boxes back to original image size
            if boxes.shape[0] > 0:
                scale_x = w / input_size[0]
                scale_y = h / input_size[1]
                
                # Scale box coordinates
                boxes[:, 0] *= scale_x  # x1
                boxes[:, 2] *= scale_x  # x2
                boxes[:, 1] *= scale_y  # y1
                boxes[:, 3] *= scale_y  # y2
                
                # Convert to integers
                boxes = boxes.astype(np.int32)
            
        elif self.model_type == "yolov5":
            # YOLOv5 specific detection - uses its own resizing
            results = self.model(frame)
            
            # Process predictions
            predictions = results.xyxy[0].cpu().numpy()  # x1, y1, x2, y2, confidence, class
            
            # Filter by confidence
            mask = predictions[:, 4] >= self.confidence_threshold
            filtered_predictions = predictions[mask]
            
            if len(filtered_predictions) == 0:
                return np.array([]), np.array([]), np.array([])
            
            # Extract boxes, labels, scores
            boxes = filtered_predictions[:, :4].astype(np.int32)  # x1, y1, x2, y2
            scores = filtered_predictions[:, 4]
            labels = filtered_predictions[:, 5].astype(np.int32)
            
        elif self.model_type == "yolov8":
            # YOLOv8 detection - uses its own resizing
            results = self.model(frame, verbose=False)
            result = results[0]  # Get first image result
            
            # Extract predictions
            boxes = []
            scores = []
            labels = []
            
            for box in result.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                
                if conf >= self.confidence_threshold:
                    boxes.append([int(x1), int(y1), int(x2, int(y2))])
                    scores.append(conf)
                    labels.append(int(cls))
                    
            boxes = np.array(boxes) if boxes else np.array([])
            scores = np.array(scores) if scores else np.array([])
            labels = np.array(labels) if labels else np.array([])
        
        # Apply NMS using CUDA if available
        if CUDA_OPS_AVAILABLE and len(boxes) > 0:
            try:
                # Convert to torch tensors on GPU
                boxes_tensor = torch.tensor(boxes, dtype=torch.float32, device=self.device)
                scores_tensor = torch.tensor(scores, dtype=torch.float32, device=self.device)
                
                # Apply NMS
                keep_indices = cuda_ops.nms(boxes_tensor, scores_tensor, 0.45)
                
                # Convert back to numpy
                keep_indices = keep_indices.cpu().numpy()
                
                # Filter the results
                boxes = boxes[keep_indices]
                scores = scores[keep_indices]
                labels = labels[keep_indices]
                
                print(f"CUDA NMS applied to {len(keep_indices)} boxes")
            except Exception as e:
                print(f"CUDA NMS failed: {e}")
        
        detection_time = (time.time() - start_time) * 1000
        print(f"Detection time: {detection_time:.1f}ms, Objects: {len(boxes)}")
        
        return boxes, labels, scores