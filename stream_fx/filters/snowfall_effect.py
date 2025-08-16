import cv2
import numpy as np
from typing import Optional, Dict, Any, List
from .base_filter import BaseFilter

class SnowfallFilter(BaseFilter):
    """
    Overlays a simulated snowfall animation on the video frame.
    """

    @property
    def identifier(self) -> str:
        return "snowfall"

    @property
    def name(self) -> str:
        return "Snowfall"

    @property
    def category(self) -> str:
        return "Animation"

    def __init__(self):
        """Initializes the filter's state."""
        self.snowflakes = None
        self.frame_height = 0
        self.frame_width = 0
        self.flake_count = 200
        self.flake_speed = 2

    def get_parameters(self) -> List[Dict[str, Any]]:
        """Declare adjustable parameters for the snow effect."""
        return [
            {
                "identifier": "flake_count",
                "name": "Flake Count",
                "type": "slider", "min": 10, "max": 1000, "step": 10, "default": 200
            },
            {
                "identifier": "flake_speed",
                "name": "Speed",
                "type": "slider", "min": 1, "max": 10, "step": 1, "default": 2
            }
        ]

    def update_parameters(self, params: Dict[str, Any]):
        """Update snow parameters and reset the animation if they change."""
        changed = False
        if "flake_count" in params and self.flake_count != int(params["flake_count"]):
            self.flake_count = int(params["flake_count"])
            changed = True
        if "flake_speed" in params and self.flake_speed != int(params["flake_speed"]):
            self.flake_speed = int(params["flake_speed"])
            changed = True
        
        if changed:
            self.snowflakes = None # Force re-initialization

    def _initialize_snowflakes(self, h, w):
        """Creates the initial random state for the snowflakes."""
        self.frame_height, self.frame_width = h, w
        # Create flakes with random x, y, radius, and speed factor
        self.snowflakes = np.random.rand(self.flake_count, 4)
        self.snowflakes[:, 0] *= w  # x position
        self.snowflakes[:, 1] *= h  # y position
        self.snowflakes[:, 2] = self.snowflakes[:, 2] * 2 + 1  # radius (1-3px)
        self.snowflakes[:, 3] = self.snowflakes[:, 3] * 0.5 + 0.5 # speed variation

    def process(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Updates snowflake positions and draws them on the frame."""
        h, w, _ = frame.shape
        if self.snowflakes is None or h != self.frame_height or w != self.frame_width:
            self._initialize_snowflakes(h, w)

        # Update snowflake positions
        self.snowflakes[:, 1] += self.snowflakes[:, 3] * self.flake_speed # Move down
        
        # Reset flakes that have moved off-screen
        off_screen_mask = self.snowflakes[:, 1] > h
        self.snowflakes[off_screen_mask, 1] = 0 # Reset y to top
        self.snowflakes[off_screen_mask, 0] = np.random.rand(np.sum(off_screen_mask)) * w # New random x

        # Create a copy of the frame to draw on
        overlay = frame.copy()
        
        # Draw each snowflake
        for x, y, radius, _ in self.snowflakes:
            cv2.circle(overlay, (int(x), int(y)), int(radius), (255, 255, 255), -1)

        # Blend the overlay with the original frame for a more subtle effect
        return cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
