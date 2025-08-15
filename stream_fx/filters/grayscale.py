import cv2
import numpy as np
from .base_filter import BaseFilter

class GrayscaleFilter(BaseFilter):
    """A simple filter that converts the frame to grayscale."""

    @property
    def identifier(self) -> str:
        return "grayscale"

    @property
    def name(self) -> str:
        return "Grayscale"

    @property
    def category(self) -> str:
        return "Basic"

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Converts the BGR frame to grayscale, then back to BGR for compatibility."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
