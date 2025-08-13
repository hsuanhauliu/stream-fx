import cv2
import numpy as np
import requests
import base64
import logging
from ..base_filter import BaseFilter

class FaceDetectionFilter(BaseFilter):
    """
    Detects faces by calling a local inference server and draws bounding boxes.
    """

    @property
    def identifier(self) -> str:
        return "face_detection"

    @property
    def name(self) -> str:
        return "Face Detection (Advanced)"

    def __init__(self):
        """Initializes the filter with the server URL."""
        self.server_url = "http://127.0.0.1:8000/predict"

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Encodes the frame, sends it to the inference server, and draws bounding boxes.
        """
        # 1. Encode the frame to a JPEG format in memory.
        _, buffer = cv2.imencode('.jpg', frame)
        
        # 2. Convert the image buffer to a Base64 string.
        base64_img = base64.b64encode(buffer).decode('utf-8')

        # 3. Prepare the JSON payload for the API request.
        payload = {"base64_imgs": [base64_img]}

        try:
            # 4. Send the POST request to the inference server with a timeout.
            response = requests.post(self.server_url, json=payload, timeout=0.5)
            response.raise_for_status() # Raise an exception for HTTP error codes.

            # 5. Parse the JSON response from the server.
            data = response.json()
            
            # 6. Check for bounding boxes and draw them on the frame.
            # The API returns a list of results for each image; we sent one image.
            if "bounding_boxes" in data and len(data["bounding_boxes"]) > 0:
                bounding_boxes = data["bounding_boxes"][0]
                
                for box in bounding_boxes:
                    # Extract coordinates for each detected face.
                    left, top, right, bottom = box['left'], box['top'], box['right'], box['bottom']
                    # Draw a green rectangle around the face.
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        except requests.exceptions.RequestException as e:
            # If the server is down or the request fails, log a debug message
            # and return the original frame without crashing the application.
            logging.debug(f"Face detection server request failed: {e}")
        
        except Exception as e:
            # Catch any other unexpected errors during processing.
            logging.error(f"An error occurred in the face detection filter: {e}")

        return frame
