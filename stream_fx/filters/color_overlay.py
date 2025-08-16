import cv2
import numpy as np
from typing import Optional, Dict, Any, List
from .base_filter import BaseFilter

class ColorOverlayFilter(BaseFilter):
    """
    Overlays a solid, adjustable color on the entire video frame.
    """

    @property
    def identifier(self) -> str:
        return "color_overlay"

    @property
    def name(self) -> str:
        return "Color Overlay"

    @property
    def category(self) -> str:
        return "Artistic"

    def __init__(self):
        """Initializes the filter with default color (red) and opacity."""
        self.blue = 0
        self.green = 0
        self.red = 255
        self.opacity = 0.5

    def get_parameters(self) -> List[Dict[str, Any]]:
        """Declare the adjustable parameters for color and opacity."""
        return [
            {
                "identifier": "red",
                "name": "Red",
                "type": "slider",
                "min": 0, "max": 255, "step": 1, "default": 255
            },
            {
                "identifier": "green",
                "name": "Green",
                "type": "slider",
                "min": 0, "max": 255, "step": 1, "default": 0
            },
            {
                "identifier": "blue",
                "name": "Blue",
                "type": "slider",
                "min": 0, "max": 255, "step": 1, "default": 0
            },
            {
                "identifier": "opacity",
                "name": "Opacity",
                "type": "slider",
                "min": 0, "max": 100, "step": 1, "default": 50
            }
        ]

    def update_parameters(self, params: Dict[str, Any]):
        """Update the color and opacity from the UI."""
        if "red" in params:
            self.red = int(params["red"])
        if "green" in params:
            self.green = int(params["green"])
        if "blue" in params:
            self.blue = int(params["blue"])
        if "opacity" in params:
            self.opacity = int(params["opacity"]) / 100.0

    def process(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Applies the color overlay to the frame."""
        # Create a new image of the same size as the frame, filled with the chosen color.
        # Note: OpenCV uses BGR color order, not RGB.
        overlay = np.full(frame.shape, (self.blue, self.green, self.red), dtype=np.uint8)

        # Blend the original frame with the color overlay.
        # The formula is: output = frame * (1 - opacity) + overlay * opacity
        return cv2.addWeighted(overlay, self.opacity, frame, 1 - self.opacity, 0)
