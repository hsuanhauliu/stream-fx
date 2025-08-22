import cv2
import mediapipe as mp
import numpy as np
import requests
import base64
import logging
import json
import threading
from typing import Dict, Any, Optional, List

# --- Optional Dependencies ---
try:
    import websockets.sync.client
except ImportError:
    websockets = None


from ..base_filter import BaseFilter

class FaceDetectionFilter(BaseFilter):
    """
    Detects faces and draws bounding boxes.
    Uses MediaPipe by default, but can be configured to use a remote inference server (HTTP/WebSocket).
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
        """Initializes the filter's state."""
        self.detection_mode = 'none' # Will be set in initialize()
        # Server-based detection
        self.server_url = None
        self.ws_url = None
        self.ws_connection = None
        self.ws_lock = threading.Lock()
        self.failure_count = 0
        self.max_failures = 5
        # MediaPipe-based detection
        self.face_detection = None

    def initialize(self, config: Dict[str, Any] = None):
        """Initializes the detection mode based on the provided configuration."""
        use_server = False
        if config:
            if 'ws_url' in config and websockets:
                self.ws_url = config['ws_url']
                self.detection_mode = 'ws'
                logging.info(f"Face detection filter using WebSocket URL: {self.ws_url}")
                use_server = True
            elif 'server_url' in config:
                self.server_url = config['server_url']
                self.detection_mode = 'http'
                logging.info(f"Face detection filter using custom HTTP URL: {self.server_url}")
                use_server = True

        if not use_server:
            if mp:
                self.detection_mode = 'mediapipe'
                self.face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
                logging.info("Face detection filter using built-in MediaPipe.")
            else:
                self.detection_mode = 'none'
                logging.warning("No server URL provided and 'mediapipe' is not installed. Face detection will be disabled.")

    def on_deactivate(self):
        """Cleans up resources when the filter is deactivated."""
        with self.ws_lock:
            if self.ws_connection:
                try:
                    self.ws_connection.close()
                    logging.info("WebSocket connection closed.")
                except Exception as e:
                    logging.error(f"Error closing WebSocket connection: {e}")
                finally:
                    self.ws_connection = None
        if self.face_detection:
            self.face_detection.close()
            self.face_detection = None

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
        """Sends detection request via WebSocket."""
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
            self.on_deactivate()
        return None

    def _detect_faces_mediapipe(self, frame: np.ndarray) -> Optional[List[Dict[str, int]]]:
        """Detects faces using the MediaPipe library."""
        if self.face_detection is None:
            self.initialize() # Re-initialize if deactivated
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        boxes = []
        if results.detections:
            h, w, _ = frame.shape
            for detection in results.detections:
                box = detection.location_data.relative_bounding_box
                boxes.append({
                    "left": int(box.xmin * w),
                    "top": int(box.ymin * h),
                    "right": int((box.xmin + box.width) * w),
                    "bottom": int((box.ymin + box.height) * h)
                })
        return boxes

    def _detect_faces(self, frame: np.ndarray) -> Optional[List[Dict[str, int]]]:
        """Dispatches to the correct detection method based on the initialized mode."""
        if self.detection_mode == 'mediapipe':
            return self._detect_faces_mediapipe(frame)
        
        # For server modes, prepare the payload
        _, buffer = cv2.imencode('.jpg', frame)
        base64_img = base64.b64encode(buffer).decode('utf-8')
        payload = {"base64_imgs": [base64_img]}

        if self.detection_mode == 'ws':
            return self._detect_faces_ws(payload)
        elif self.detection_mode == 'http':
            return self._detect_faces_http(payload)
        
        return None # No valid detection mode

    def process(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detects faces and draws bounding boxes on the frame."""
        bounding_boxes = self._detect_faces(frame)
        
        if bounding_boxes is not None:
            for box in bounding_boxes:
                left, top, right, bottom = map(int, [box['left'], box['top'], box['right'], box['bottom']])
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        if self.detection_mode in ['http', 'ws'] and self.failure_count >= self.max_failures:
            logging.warning(f"Face detection filter failed {self.max_failures} consecutive times. Disabling.")
            self.failure_count = 0
            return None

        return frame
