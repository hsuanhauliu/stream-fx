import cv2
import numpy as np
from .base_filter import BaseFilter
from typing import Dict, Any


class BlurBackgroundFilter(BaseFilter):
    """A filter that detects people and blurs the background."""

    def __init__(self):
        self.hog = None

    @property
    def identifier(self) -> str:
        return "blur"

    @property
    def name(self) -> str:
        return "Blur BG"

    def initialize(self, config: Dict[str, Any] = None):
        """Initializes the HOG person detector."""
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Detects humans and applies a blur to the background."""
        if self.hog is None:
            # Return original frame if HOG is not initialized
            return frame

        (rects, weights) = self.hog.detectMultiScale(
            frame, winStride=(4, 4), padding=(8, 8), scale=1.05
        )

        if len(rects) == 0:
            return frame  # Return original if no one is detected

        mask = np.zeros(frame.shape[:2], dtype="uint8")
        for x, y, w, h in rects:
            pad_w, pad_h = int(0.15 * w), int(0.05 * h)
            cv2.rectangle(
                mask, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), 255, -1
            )

        blurred = cv2.GaussianBlur(frame, (51, 51), 0)
        output = np.where(mask[..., None].astype(bool), frame, blurred)
        return output
