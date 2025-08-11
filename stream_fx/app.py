import cv2
import numpy as np
import argparse
import signal
import sys
import time
import logging
import io
import threading
import uvicorn
from fastapi import FastAPI
from starlette.responses import StreamingResponse

# --- FastAPI App and Global variables ---
app = FastAPI()
output_frame = None
lock = threading.Lock()

async def frame_generator():
    """A generator function that yields JPEG frames for streaming."""
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                time.sleep(0.01) # wait for a frame to be available
                continue
            # Encode frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', output_frame)
            if not ret:
                continue
            frame_data = jpeg.tobytes()
        
        # Yield the frame in the format required for multipart/x-mixed-replace
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    """The main video streaming endpoint."""
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=--frame")

# --- Application Logic ---
running = True

def signal_handler(sig, frame):
    """Handles the Ctrl+C signal to gracefully exit the application."""
    global running
    logging.info("Ctrl+C detected. Exiting gracefully...")
    running = False

def parse_arguments():
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process camera frames and stream them over HTTP for OBS using FastAPI."
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode to visualize frames and set logging to DEBUG level.'
    )
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host for the streaming server.')
    parser.add_argument('--port', type=int, default=8080, help='Port for the streaming server.')
    return parser.parse_args()

def initialize_camera():
    """Initializes the camera, gets its properties, and returns them."""
    logging.info("Attempting to open the default camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not start camera. Is it connected and not in use?")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        logging.warning("Camera does not provide FPS. Defaulting to 30.")
        fps = 30

    logging.info(f"Camera opened successfully: {width}x{height} @ {fps} FPS")
    return cap, width, height, fps

def apply_grayscale_effect(frame):
    """Applies a grayscale effect to an image frame."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def handle_debug_input(processing_enabled):
    """Checks for user input keys 'p' and 'q' and updates state."""
    global running
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        logging.info("'q' key pressed. Exiting...")
        running = False
    elif key == ord('p'):
        processing_enabled = not processing_enabled
        status = "ON" if processing_enabled else "OFF"
        logging.info(f"Processing toggled {status}")
    return processing_enabled

def main():
    """
    Main function to run the camera processing and the FastAPI streaming server.
    """
    global output_frame, lock, running
    
    args = parse_arguments()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    signal.signal(signal.SIGINT, signal_handler)
    cap = None
    
    # Configure and start Uvicorn server in a background thread
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run)
    server_thread.daemon = True
    server_thread.start()
    logging.info(f"Streaming server started at http://{args.host}:{args.port}/video_feed")

    try:
        cap, width, height, fps = initialize_camera()
        
        if args.debug:
            logging.info("DEBUG MODE: Press 'p' to toggle processing, 'q' to quit.")

        processing_enabled = True
        
        while running:
            ret, frame = cap.read()
            if not ret:
                logging.error("Can't receive frame (stream end?). Exiting...")
                break

            frame_to_display = frame
            if processing_enabled:
                processed_frame = apply_grayscale_effect(frame)
                frame_to_stream = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
                if args.debug:
                    frame_to_display = processed_frame
            else:
                frame_to_stream = frame
                if args.debug:
                    frame_to_display = frame

            # Update the global output frame for the server
            with lock:
                output_frame = frame_to_stream.copy()

            if args.debug:
                cv2.imshow('Original Camera Feed', frame)
                cv2.imshow('Processed Feed', frame_to_display)
                processing_enabled = handle_debug_input(processing_enabled)
            
            # Control frame rate
            time.sleep(1/fps)

    except (RuntimeError, BrokenPipeError) as e:
        logging.error(f"An error occurred: {e}")
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt detected. Exiting.")
    finally:
        running = False
        # The server thread is a daemon, so it will exit automatically.
        logging.info("Cleaning up resources...")
        if cap and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        logging.info("Cleanup complete. Application terminated.")
        sys.exit(0)

if __name__ == "__main__":
    main()
