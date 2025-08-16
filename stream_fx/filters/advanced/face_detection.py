import cv2
import numpy as np
import requests
import base64
import logging
import json
import threading
from typing import Dict, Any, Optional, List

# The 'websockets' library is required for this filter.
# Please ensure it is added to your requirements.txt: `websockets`
try:
    import websockets.sync.client
except ImportError:
    logging.warning("The 'websockets' library is not installed. WebSocket support will be disabled.")
    websockets = None

from ..base_filter import BaseFilter

class FaceDetectionFilter(BaseFilter):
    """
    Detects faces by calling a local inference server and draws bounding boxes.
    Supports WebSocket for efficient communication, with an HTTP fallback.
    """

    @property
    def identifier(self) -> str:
        return "face_detection"

    @property
    def name(self) -> str:
        return "Face Detection"

    @property
    def category(self) -> str:
        return "Face"

    def __init__(self):
        """Initializes the filter with default URLs and connection state."""
        self.server_url = "http://127.0.0.1:8000/predict"
        self.ws_url = None
        self.ws_connection = None
        self.ws_lock = threading.Lock()
        self.failure_count = 0
        self.max_failures = 5

    def initialize(self, config: Dict[str, Any] = None):
        """
        Overrides the server URLs if they are provided in the config.
        """
        if config:
            if 'server_url' in config:
                self.server_url = config['server_url']
                logging.info(f"Face detection filter using custom HTTP URL: {self.server_url}")
            if 'ws_url' in config:
                if websockets:
                    self.ws_url = config['ws_url']
                    logging.info(f"Face detection filter using WebSocket URL: {self.ws_url}")
                else:
                    logging.warning("WebSocket URL provided in config, but 'websockets' library is not installed. Ignoring.")

    def on_deactivate(self):
        """
        Called when the filter is removed from the active stack.
        Closes any open WebSocket connection.
        """
        with self.ws_lock:
            if self.ws_connection:
                try:
                    self.ws_connection.close()
                    logging.info("WebSocket connection closed.")
                except Exception as e:
                    logging.error(f"Error closing WebSocket connection: {e}")
                finally:
                    self.ws_connection = None

    def _detect_faces_http(self, payload: Dict) -> Optional[List[Dict[str, int]]]:
        """Sends detection request via HTTP POST."""
        try:
            response = requests.post(self.server_url, json=payload, timeout=0.5)
            response.raise_for_status()
            self.failure_count = 0
            data = response.json()
            if "bounding_boxes" in data and len(data["bounding_boxes"]) > 0:
                return data["bounding_boxes"][0]
        except requests.exceptions.RequestException as e:
            self.failure_count += 1
            logging.debug(f"HTTP request failed ({self.failure_count}/{self.max_failures}): {e}")
        return None

    def _detect_faces_ws(self, payload: Dict) -> Optional[List[Dict[str, int]]]:
        """Sends detection request via WebSocket, establishing connection if needed."""
        try:
            with self.ws_lock:
                if self.ws_connection is None:
                    self.ws_connection = websockets.sync.client.connect(self.ws_url)
                    logging.info("WebSocket connection established.")
            
            self.ws_connection.send(json.dumps(payload))
            response_str = self.ws_connection.recv(timeout=0.5)
            data = json.loads(response_str)
            self.failure_count = 0
            if "bounding_boxes" in data and len(data["bounding_boxes"]) > 0:
                return data["bounding_boxes"][0]
        except Exception as e:
            logging.debug(f"WebSocket communication error: {e}")
            self.failure_count += 1
            self.on_deactivate() # Close the connection on error
        return None

    def _detect_faces(self, frame: np.ndarray) -> Optional[List[Dict[str, int]]]:
        """
        Calls the inference server to detect faces and returns a list of bounding boxes.
        """
        _, buffer = cv2.imencode('.jpg', frame)
        base64_img = base64.b64encode(buffer).decode('utf-8')
        payload = {"base64_imgs": [base64_img]}

        try:
            if self.ws_url:
                return self._detect_faces_ws(payload)
            else:
                return self._detect_faces_http(payload)
        except Exception as e:
            self.failure_count += 1
            logging.error(f"An unexpected error occurred during face detection ({self.failure_count}/{self.max_failures}): {e}")
            return None

    def process(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detects faces and draws bounding boxes on the frame.
        If it fails consecutively, it will disable itself.
        """
        bounding_boxes = self._detect_faces(frame)
        
        if bounding_boxes is not None:
            for box in bounding_boxes:
                left, top, right, bottom = map(int, [box['left'], box['top'], box['right'], box['bottom']])
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        if self.failure_count >= self.max_failures:
            logging.warning(f"Face detection filter failed {self.max_failures} consecutive times. Disabling.")
            self.failure_count = 0
            return None

        return frame
