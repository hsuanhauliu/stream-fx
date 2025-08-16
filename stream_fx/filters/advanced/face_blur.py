import cv2
import numpy as np
from typing import Optional
from .face_detection import FaceDetectionFilter

class FaceBlurFilter(FaceDetectionFilter):
    """
    Inherits from FaceDetectionFilter to detect faces and then blurs them.
    """

    @property
    def identifier(self) -> str:
        return "face_blur"

    @property
    def name(self) -> str:
        return "Face Blur"

    def process(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detects faces using the parent method and applies a blur to each bounding box.
        """
        # Call the parent's detection method to get bounding boxes
        bounding_boxes = self._detect_faces(frame)
        
        if bounding_boxes is not None:
            for box in bounding_boxes:
                left = int(box['left'])
                top = int(box['top'])
                right = int(box['right'])
                bottom = int(box['bottom'])

                # Ensure the coordinates are within the frame boundaries
                if left < right and top < bottom:
                    # Extract the region of interest (the face)
                    face_roi = frame[top:bottom, left:right]
                    
                    # Apply a strong Gaussian blur to the face region
                    blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
                    
                    # Replace the original face area with the blurred one
                    frame[top:bottom, left:right] = blurred_face

        # Check for failures and self-disable if necessary (logic from parent)
        if self.failure_count >= self.max_failures:
            self.failure_count = 0 
            return None

        return frame
