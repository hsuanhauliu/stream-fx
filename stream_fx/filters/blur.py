import cv2
import numpy as np
from typing import Dict, Any, List
from .base_filter import BaseFilter

class BlurBackgroundFilter(BaseFilter):
    """A filter that detects people and blurs the background."""

    def __init__(self):
        self.hog = None
        # Default blur intensity. Must be an odd number.
        self.blur_intensity = 51

    @property
    def identifier(self) -> str:
        return "blur"

    @property
    def name(self) -> str:
        return "Blur BG"

    @property
    def category(self) -> str:
        return "Background"

    def get_parameters(self) -> List[Dict[str, Any]]:
        """Declare the adjustable blur intensity parameter."""
        return [
            {
                "identifier": "blur_intensity",
                "name": "Blur Intensity",
                "type": "slider",
                "min": 1,
                "max": 99,
                "step": 2, # Step by 2 to ensure the value is always odd
                "default": self.blur_intensity
            }
        ]

    def update_parameters(self, params: Dict[str, Any]):
        """Update the blur intensity from the UI."""
        if "blur_intensity" in params:
            # Ensure the value is an odd integer
            self.blur_intensity = int(params["blur_intensity"])
            if self.blur_intensity % 2 == 0:
                self.blur_intensity += 1

    def initialize(self, config: Dict[str, Any] = None):
        """Initializes the HOG person detector."""
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Detects humans and applies a blur to the background."""
        if self.hog is None:
            return frame
            
        (rects, weights) = self.hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

        if len(rects) == 0:
            return frame

        mask = np.zeros(frame.shape[:2], dtype="uint8")
        for (x, y, w, h) in rects:
            pad_w, pad_h = int(0.15 * w), int(0.05 * h)
            cv2.rectangle(mask, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), 255, -1)

        # Use the current blur_intensity value
        blurred = cv2.GaussianBlur(frame, (self.blur_intensity, self.blur_intensity), 0)
        
        output = np.where(mask[...,None].astype(bool), frame, blurred)
        return output
