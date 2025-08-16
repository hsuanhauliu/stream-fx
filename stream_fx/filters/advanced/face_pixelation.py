import cv2
import numpy as np
from typing import Optional, List, Dict, Any
from .face_detection import FaceDetectionFilter

class FacePixelationFilter(FaceDetectionFilter):
    """
    Inherits from FaceDetectionFilter to detect faces and then pixelates them.
    """

    @property
    def identifier(self) -> str:
        return "face_pixelation"

    @property
    def name(self) -> str:
        return "Face Pixelation"

    def __init__(self):
        super().__init__()
        self.pixel_size = 15

    def get_parameters(self) -> List[Dict[str, Any]]:
        """Declare the adjustable pixelation block size."""
        return [
            {
                "identifier": "pixel_size",
                "name": "Pixel Size",
                "type": "slider",
                "min": 3, "max": 49, "step": 2, "default": 15
            }
        ]

    def update_parameters(self, params: Dict[str, Any]):
        """Update the pixel size from the UI."""
        if "pixel_size" in params:
            self.pixel_size = int(params["pixel_size"])

    def process(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detects faces using the parent method and applies pixelation to each bounding box.
        """
        bounding_boxes = self._detect_faces(frame)
        
        if bounding_boxes is not None:
            for box in bounding_boxes:
                left, top, right, bottom = map(int, [box['left'], box['top'], box['right'], box['bottom']])

                if left < right and top < bottom:
                    # Extract the region of interest (the face)
                    face_roi = frame[top:bottom, left:right]
                    
                    # Get the dimensions of the face ROI
                    h, w, _ = face_roi.shape
                    
                    # Resize the ROI to a small size, then resize it back to the original size.
                    # This creates the pixelation effect.
                    temp = cv2.resize(face_roi, (self.pixel_size, self.pixel_size), interpolation=cv2.INTER_LINEAR)
                    pixelated_face = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    # Replace the original face area with the pixelated one
                    frame[top:bottom, left:right] = pixelated_face

        if self.failure_count >= self.max_failures:
            self.failure_count = 0 
            return None

        return frame
