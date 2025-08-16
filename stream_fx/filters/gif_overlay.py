import cv2
import numpy as np
import os
import logging
import time
from typing import Optional, Dict, Any, List

# This filter requires the 'imageio' library.
# It will be skipped if the library is not installed.
try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

from .base_filter import BaseFilter

class GifOverlayFilter(BaseFilter):
    """
    Overlays an animated GIF on the video frame.
    """

    @property
    def identifier(self) -> str:
        return "gif_overlay"

    @property
    def name(self) -> str:
        return "GIF Overlay"

    @property
    def category(self) -> str:
        return "Overlay"

    def __init__(self):
        """Initializes the filter's state."""
        if imageio is None:
            raise ImportError("The 'imageio' library is required for the GIF filter.")
        
        self.gif_frames = []
        self.current_frame_index = 0
        self.last_frame_time = 0
        self.frame_duration = 0.1  # Default duration if not found in GIF metadata
        self.loop = True
        self.animation_finished = False
        self.location = "Top Right" # Default location

    def initialize(self, config: Dict[str, Any] = None):
        """Loads the GIF from the path specified in the config."""
        if not (config and 'gif_path' in config):
            logging.warning("No 'gif_path' specified in config for GIF Overlay filter. It will not run.")
            return

        gif_path = os.path.abspath(config['gif_path'])
        if not os.path.exists(gif_path):
            logging.warning(f"GIF file not found at '{gif_path}'.")
            return

        try:
            # Read all frames of the GIF
            self.gif_frames = imageio.mimread(gif_path, memtest=False)
            
            # Get metadata to determine frame duration
            gif_reader = imageio.get_reader(gif_path)
            meta = gif_reader.get_meta_data()
            self.frame_duration = meta.get('duration', 100) / 1000.0 # Duration in seconds
            
            # Convert frames from RGB (imageio) to BGR (OpenCV) and add alpha channel if missing
            processed_frames = []
            for frame in self.gif_frames:
                if frame.shape[2] == 3: # If no alpha channel
                    frame_rgba = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                else:
                    frame_rgba = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
                processed_frames.append(frame_rgba)
            self.gif_frames = processed_frames

            logging.info(f"Loaded {len(self.gif_frames)} frames from '{gif_path}' with a duration of {self.frame_duration:.3f}s per frame.")
        except Exception as e:
            logging.error(f"Failed to load GIF file: {e}")
            self.gif_frames = []

    def get_parameters(self) -> List[Dict[str, Any]]:
        """Declare the playback mode and location parameters."""
        return [
            {
                "identifier": "playback_mode",
                "name": "Playback",
                "type": "select",
                "options": ["Loop", "Play Once"],
                "default": "Loop"
            },
            {
                "identifier": "location",
                "name": "Location",
                "type": "select",
                "options": ["Top Right", "Top Left", "Bottom Left", "Bottom Right", "Center"],
                "default": "Top Right"
            }
        ]

    def update_parameters(self, params: Dict[str, Any]):
        """Update parameters from the UI."""
        if "playback_mode" in params:
            self.loop = (params["playback_mode"] == "Loop")
            # Reset animation if mode changes
            self.on_deactivate()
        if "location" in params:
            self.location = params["location"]

    def on_deactivate(self):
        """Resets the animation when the filter is deactivated."""
        self.current_frame_index = 0
        self.last_frame_time = 0
        self.animation_finished = False

    def process(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Overlays the current GIF frame onto the main video frame."""
        if not self.gif_frames:
            return frame

        # If the animation has finished in "Play Once" mode, disable the filter.
        if self.animation_finished and not self.loop:
            self.on_deactivate() # Reset for the next time it's activated.
            return None

        # Advance to the next frame of the GIF based on elapsed time
        current_time = time.time()
        if self.last_frame_time == 0:
            self.last_frame_time = current_time

        if (current_time - self.last_frame_time) > self.frame_duration:
            next_frame_index = self.current_frame_index + 1
            if self.loop:
                self.current_frame_index = next_frame_index % len(self.gif_frames)
            elif next_frame_index < len(self.gif_frames):
                self.current_frame_index = next_frame_index
            else:
                # Animation is done. The last frame will be displayed this one time.
                # The `animation_finished` flag will trigger removal on the next call.
                self.animation_finished = True
            
            self.last_frame_time = current_time

        # Get the current GIF frame to overlay
        gif_frame = self.gif_frames[self.current_frame_index]

        # Define where to place the GIF
        h, w, _ = frame.shape
        gh, gw, _ = gif_frame.shape
        
        if gh > h or gw > w:
            return frame

        # Calculate offsets based on location
        if self.location == "Top Left":
            x_offset, y_offset = 0, 0
        elif self.location == "Bottom Left":
            x_offset, y_offset = 0, h - gh
        elif self.location == "Bottom Right":
            x_offset, y_offset = w - gw, h - gh
        elif self.location == "Center":
            x_offset, y_offset = (w - gw) // 2, (h - gh) // 2
        else: # Default to Top Right
            x_offset, y_offset = w - gw, 0

        # Create the Region of Interest (ROI)
        roi = frame[y_offset:y_offset+gh, x_offset:x_offset+gw]

        # Blend the GIF frame onto the ROI using its alpha channel
        alpha_mask = gif_frame[:, :, 3] / 255.0
        inverse_alpha_mask = 1.0 - alpha_mask
        
        for c in range(0, 3):
            roi[:, :, c] = (alpha_mask * gif_frame[:, :, c] +
                            inverse_alpha_mask * roi[:, :, c])

        return frame
