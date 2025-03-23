import cv2

def start_webcam(camera_id=0, width=640, height=480):
    """
    Initialize the webcam
    
    Args:
        camera_id: Camera device ID
        width: Desired frame width
        height: Desired frame height
        
    Returns:
        Initialized webcam capture object
    """
    cap = cv2.VideoCapture(camera_id)
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    if not cap.isOpened():
        raise Exception("Could not open webcam")
        
    print(f"Webcam initialized with resolution {width}x{height}")
    return cap

def get_frame(cap):
    """
    Get a frame from the webcam
    
    Args:
        cap: Webcam capture object
        
    Returns:
        Frame as numpy array
    """
    ret, frame = cap.read()
    
    if not ret:
        raise Exception("Could not read frame from webcam")
        
    return frame

def release_webcam(cap):
    """
    Release the webcam resources
    
    Args:
        cap: Webcam capture object
    """
    cap.release()
    cv2.destroyAllWindows()