import os
import time

# Import PyTorch - this is required for CUDA
try:
    import torch
    TORCH_AVAILABLE = True
    
    # Check CUDA availability
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("CUDA is not available on this system.")
        raise ImportError("CUDA is required but not available")
except ImportError as e:
    print(f"Error importing PyTorch or CUDA: {e}")
    print("This application requires PyTorch with CUDA support.")
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    # We'll exit in the main module

# Flag for CUDA operations
CUDA_OPS_AVAILABLE = False

# Check if half-precision (FP16) is supported
FP16_SUPPORTED = False
if TORCH_AVAILABLE and CUDA_AVAILABLE:
    try:
        # Create a test tensor and convert to half precision
        test_tensor = torch.zeros(1, 1, 1, 1, device='cuda')
        test_half = test_tensor.half()
        # Try a simple operation
        test_result = test_half * 2
        FP16_SUPPORTED = True
        print("Half-precision (FP16) operations are supported")
    except Exception as e:
        print(f"Half-precision not fully supported: {e}")
        print("Using full precision (FP32) only")

# Only proceed with CUDA operations if torch and CUDA are available
if TORCH_AVAILABLE and CUDA_AVAILABLE:
    # Set flag that CUDA ops are available
    CUDA_OPS_AVAILABLE = True
    
    # Create an optimized class for CUDA operations
    class CudaOps:
        def __init__(self):
            # Use torchvision's NMS which is highly optimized
            try:
                from torchvision.ops import nms as torchvision_nms
                self._nms_impl = torchvision_nms
                print("Using torchvision's optimized NMS")
            except (ImportError, AttributeError):
                print("Torchvision NMS not available, using PyTorch's batched_nms")
                # Fallback to PyTorch's implementation
                try:
                    from torchvision.ops import nms
                    self._nms_impl = nms
                    print("Using PyTorch's native NMS")
                except (ImportError, AttributeError):
                    # Final fallback to a simple implementation
                    self._nms_impl = self._nms_simple
                    print("Using simple NMS implementation")
        
        @staticmethod
        def _nms_simple(boxes, scores, threshold):
            """Simple NMS implementation"""
            keep = []
            idxs = scores.argsort(descending=True)
            
            while idxs.numel() > 0:
                # Pick the highest scoring box
                i = idxs[0].item()
                keep.append(i)
                
                # If only one box left, we're done
                if idxs.size(0) == 1:
                    break
                
                # Get IoU between top box and remaining boxes
                xx1 = torch.max(boxes[i, 0], boxes[idxs[1:], 0])
                yy1 = torch.max(boxes[i, 1], boxes[idxs[1:], 1])
                xx2 = torch.min(boxes[i, 2], boxes[idxs[1:], 2])
                yy2 = torch.min(boxes[i, 3], boxes[idxs[1:], 3])
                
                w = torch.clamp(xx2 - xx1, min=0.0)
                h = torch.clamp(yy2 - yy1, min=0.0)  # Fixed height calculation
                
                inter = w * h
                area1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
                area2 = (boxes[idxs[1:], 2] - boxes[idxs[1:], 0]) * (boxes[idxs[1:], 3] - boxes[idxs[1:], 1])
                
                # Calculate IoU
                iou = inter / (area1 + area2 - inter)
                
                # Keep boxes with IoU less than threshold
                idxs = idxs[1:][iou <= threshold]
                
            return torch.tensor(keep, device=boxes.device)
        
        def nms(self, boxes, scores, threshold=0.45):
            """Perform Non-Maximum Suppression"""
            if not boxes.is_cuda:
                boxes = boxes.cuda()
            if not scores.is_cuda:
                scores = scores.cuda()
                
            # Apply memory-efficient operations
            torch.cuda.empty_cache()  # Clear GPU memory cache
            
            # Run the optimized NMS (avoid mixed precision which can cause issues)
            result = self._nms_impl(boxes, scores, threshold)
            
            return result
    
    # Create an instance
    cuda_ops = CudaOps()
    print("CUDA operations initialized successfully")
else:
    print("CUDA operations are not available")
    
    # Create a dummy class to avoid import errors
    class CudaOps:
        @staticmethod
        def nms(boxes, scores, threshold=0.5):
            raise RuntimeError("CUDA operations are not available")
    
    # Create an instance
    cuda_ops = CudaOps()