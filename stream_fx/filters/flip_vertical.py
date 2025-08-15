import cv2
import numpy as np
from .base_filter import BaseFilter

class FlipVerticalFilter(BaseFilter):
    """A filter that flips the video frame vertically."""

    @property
    def identifier(self) -> str:
        return "flip_vertical"

    @property
    def name(self) -> str:
        return "Flip Vertical"

    @property
    def category(self) -> str:
        return "Basic"

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Flips the frame vertically."""
        return cv2.flip(frame, 0)
