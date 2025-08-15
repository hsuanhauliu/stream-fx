import cv2
import numpy as np
from .base_filter import BaseFilter

class FlipHorizontalFilter(BaseFilter):
    """A filter that flips the video frame horizontally."""

    @property
    def identifier(self) -> str:
        return "flip_horizontal"

    @property
    def name(self) -> str:
        return "Flip Horizontal"

    @property
    def category(self) -> str:
        return "Basic"

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Flips the frame horizontally."""
        return cv2.flip(frame, 1)
