import cv2
import numpy as np
import os
import logging
from typing import Optional, Dict, Any, List
from .face_detection import FaceDetectionFilter

class FaceEmojiFilter(FaceDetectionFilter):
    """
    Inherits from FaceDetectionFilter to detect faces and then overlays a
    user-selectable image on top of each bounding box.
    """

    @property
    def identifier(self) -> str:
        return "face_emoji"

    @property
    def name(self) -> str:
        return "Face Emoji (Advanced)"

    def __init__(self):
        """Initializes the filter and prepares to load overlay images."""
        super().__init__()
        self.overlay_images = {}
        self.selected_emoji = 'none' # Default to 'none'
        self.emoji_folder_path = None # No default path

    def _generate_noise_image(self) -> np.ndarray:
        """Creates a 100x100px noise image with an alpha channel as a fallback."""
        logging.warning("Generating fallback noise image for Face Emoji filter.")
        noise = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        alpha = np.full((100, 100, 1), 255, dtype=np.uint8)
        return np.concatenate([noise, alpha], axis=2)

    def initialize(self, config: Dict[str, Any] = None):
        """Loads all emoji images from the folder specified in the config."""
        super().initialize(config) # Call parent's initialize
        if config and 'emoji_folder_path' in config:
            self.emoji_folder_path = config['emoji_folder_path']
        
        if not self.emoji_folder_path:
            logging.warning("No 'emoji_folder_path' specified in config for Face Emoji filter.")
            self.overlay_images['noise'] = self._generate_noise_image()
            self.selected_emoji = 'noise'
            return

        # Load images from the specified path, relative to the current working directory
        full_path = os.path.abspath(self.emoji_folder_path)

        if not os.path.isdir(full_path):
            logging.warning(f"Emoji folder not found at '{full_path}'. The Face Emoji filter will use a fallback.")
            self.overlay_images['noise'] = self._generate_noise_image()
            self.selected_emoji = 'noise'
            return

        for filename in os.listdir(full_path):
            if filename.endswith(".png"):
                try:
                    image_path = os.path.join(full_path, filename)
                    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                    if image is not None and image.shape[2] == 4: # Ensure image has alpha channel
                        emoji_name = os.path.splitext(filename)[0]
                        self.overlay_images[emoji_name] = image
                except Exception as e:
                    logging.error(f"Error loading emoji image {filename}: {e}")
        
        if not self.overlay_images:
            logging.warning(f"No valid .png images with alpha channel found in '{full_path}'. Using fallback.")
            self.overlay_images['noise'] = self._generate_noise_image()
            self.selected_emoji = 'noise'
        else:
            logging.info(f"Loaded emojis: {list(self.overlay_images.keys())}")
            # Ensure default is 'none' even after loading images
            self.selected_emoji = 'none'


    def get_parameters(self) -> List[Dict[str, Any]]:
        """Declare the selectable emoji parameter."""
        if not self.overlay_images:
            return []
            
        # Create a list of options, starting with 'none'
        options = ['none'] + list(self.overlay_images.keys())

        return [
            {
                "identifier": "selected_emoji",
                "name": "Emoji",
                "type": "select",
                "options": options,
                "default": self.selected_emoji
            }
        ]

    def update_parameters(self, params: Dict[str, Any]):
        """Update the selected emoji from the UI."""
        if "selected_emoji" in params:
            selected = params["selected_emoji"]
            if selected == 'none' or selected in self.overlay_images:
                self.selected_emoji = selected

    def process(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detects faces and overlays the selected emoji on each bounding box.
        """
        # If 'none' is selected, or if the selection is invalid, do nothing.
        if not self.selected_emoji or self.selected_emoji == 'none' or self.selected_emoji not in self.overlay_images:
            return frame

        overlay_image = self.overlay_images[self.selected_emoji]
        bounding_boxes = self._detect_faces(frame)
        
        if bounding_boxes is not None:
            for box in bounding_boxes:
                left, top, right, bottom = map(int, [box['left'], box['top'], box['right'], box['bottom']])

                if left < right and top < bottom:
                    roi = frame[top:bottom, left:right]
                    resized_overlay = cv2.resize(overlay_image, (roi.shape[1], roi.shape[0]))
                    
                    overlay_rgb = resized_overlay[:,:,:3]
                    alpha_mask = resized_overlay[:,:,3] / 255.0
                    inverse_alpha_mask = 1.0 - alpha_mask

                    for c in range(0, 3):
                        roi[:,:,c] = (alpha_mask * overlay_rgb[:,:,c] +
                                      inverse_alpha_mask * roi[:,:,c])

        if self.failure_count >= self.max_failures:
            self.failure_count = 0 
            return None

        return frame
