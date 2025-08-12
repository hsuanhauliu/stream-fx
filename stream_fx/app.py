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
from starlette.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict
import os
import importlib
from queue import Queue

# --- FIX: Modify sys.path BEFORE attempting to import from the filters module ---
# This ensures that the 'filters' directory is recognized as a package.
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
except NameError:
    # Fallback for environments where __file__ is not defined
    if "." not in sys.path:
        sys.path.insert(0, ".")

from filters.base_filter import BaseFilter

# --- OPTIMIZATION: Camera Capture Thread ---
class CameraThread(threading.Thread):
    """A dedicated thread to read frames from the camera."""
    def __init__(self, camera_index, frame_queue):
        super().__init__()
        self.camera_index = camera_index
        self.frame_queue = frame_queue
        self.cap = None
        self.running = False
        self.daemon = True

    def run(self):
        logging.info(f"Camera thread starting for camera index {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            logging.error(f"Could not start camera at index {self.camera_index} in thread.")
            return

        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logging.warning("Camera thread: failed to grab frame.")
                time.sleep(0.1) # Wait a bit before retrying
                continue
            
            # If the queue is full, discard the old frame and put the new one.
            if self.frame_queue.full():
                self.frame_queue.get_nowait()
            self.frame_queue.put(frame)
        
        self.cap.release()
        logging.info("Camera thread stopped.")

    def stop(self):
        self.running = False

# --- Plugin Loading ---
def load_plugins() -> Dict[str, BaseFilter]:
    """Dynamically loads all filter plugins from the 'filters' directory."""
    plugins = {}
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filters_dir_path = os.path.join(script_dir, "filters")
    except NameError:
        filters_dir_path = "filters"

    logging.info(f"Searching for plugins in '{filters_dir_path}' directory...")
    if not os.path.isdir(filters_dir_path):
        logging.error(f"Filters directory not found at '{filters_dir_path}'. Please ensure it exists.")
        return plugins

    for filename in os.listdir(filters_dir_path):
        if filename.endswith(".py") and not filename.startswith("__") and filename != "base_filter.py":
            module_name = f"filters.{filename[:-3]}"
            try:
                module = importlib.import_module(module_name)
                for item_name in dir(module):
                    item = getattr(module, item_name)
                    if isinstance(item, type) and issubclass(item, BaseFilter) and item is not BaseFilter:
                        plugin_instance = item()
                        plugin_instance.initialize() # Call optional initializer
                        plugins[plugin_instance.identifier] = plugin_instance
                        logging.info(f"Successfully loaded plugin: '{plugin_instance.name}' ({plugin_instance.identifier})")
            except Exception as e:
                logging.error(f"Failed to load plugin from {filename}: {e}")
    return plugins

# --- FastAPI App and Global variables ---
app = FastAPI()
output_frame = None
lock = threading.Lock()
loaded_plugins = load_plugins()

# --- Global State Management ---
state = {
    "active_effects": [], # A list of effect identifiers, defining the stack
    "effects_enabled": True # Master toggle for the entire stack
}
state_lock = threading.Lock()

# --- API Models ---
class FilterInfo(BaseModel):
    identifier: str
    name: str

class ControlStatus(BaseModel):
    active_effects: List[str]
    available_filters: List[FilterInfo]
    effects_enabled: bool

class SetStackRequest(BaseModel):
    effects: List[str] = Field(..., description="An ordered list of effect identifiers.")


# --- Video Streaming Endpoint ---
async def frame_generator():
    """A generator function that yields JPEG frames for streaming."""
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                time.sleep(0.01)
                continue
            ret, jpeg = cv2.imencode('.jpg', output_frame)
            if not ret:
                continue
            frame_data = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    """The main video streaming endpoint."""
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=--frame")

# --- Control API Endpoints ---
@app.get("/control/status", response_model=ControlStatus)
async def get_status():
    """Returns the list of available filters and the active stack status."""
    with state_lock:
        active_effects = state["active_effects"]
        effects_enabled = state["effects_enabled"]
    
    available = [
        {"identifier": plugin.identifier, "name": plugin.name}
        for plugin in loaded_plugins.values()
    ]
    return {"active_effects": active_effects, "available_filters": available, "effects_enabled": effects_enabled}

@app.post("/control/set_stack", response_model=Dict)
async def set_stack(request: SetStackRequest):
    """Sets the active filter stack to a new configuration."""
    with state_lock:
        # Validate that all requested effects are valid plugins
        valid_effects = [eff for eff in request.effects if eff in loaded_plugins]
        state["active_effects"] = valid_effects
        logging.info(f"Filter stack updated via API: {valid_effects}")
        return {"status": "success", "active_effects": valid_effects}

@app.post("/control/toggle_all", response_model=Dict)
async def toggle_all_effects():
    """Toggles the entire filter stack on or off (disable)."""
    with state_lock:
        state["effects_enabled"] = not state["effects_enabled"]
        logging.info(f"All effects toggled via API. Enabled: {state['effects_enabled']}")
        return {"status": "success", "effects_enabled": state["effects_enabled"]}

# --- Web UI Endpoint ---
@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """Serves the control UI from the static directory."""
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    ui_path = os.path.join(static_dir, "control_panel.html")
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    else:
        return HTMLResponse("<h1>Error: control_panel.html not found in static directory.</h1>", status_code=404)

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
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host for the streaming server.')
    parser.add_argument('--port', type=int, default=8080, help='Port for the streaming server.')
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Index of the camera to use (e.g., 0, 1, 2).'
    )
    return parser.parse_args()

def handle_debug_input():
    """Checks for user input keys 'q' to quit."""
    global running
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        logging.info("'q' key pressed. Exiting...")
        running = False

def main():
    """
    Main function to run the camera processing and the FastAPI streaming server.
    """
    global output_frame, lock, running, state, state_lock
    
    args = parse_arguments()
    
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    signal.signal(signal.SIGINT, signal_handler)
    
    # --- FastAPI Server Thread ---
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run)
    server_thread.daemon = True
    server_thread.start()
    logging.info(f"Streaming server started at http://{args.host}:{args.port}/video_feed")
    logging.info(f"Control UI available at http://{args.host}:{args.port}/")

    # --- Camera and Processing Thread ---
    # A queue of size 1 will always hold the latest frame
    frame_queue = Queue(maxsize=1)
    camera_thread = CameraThread(args.camera, frame_queue)
    camera_thread.start()

    try:
        if args.debug:
            logging.info("DEBUG MODE: Press 'q' to quit.")
        
        while running:
            try:
                # Get the latest frame from the camera thread
                frame = frame_queue.get(timeout=1)
            except Exception:
                # If queue is empty for 1 second, check if we should still be running
                continue

            with state_lock:
                active_stack = state["active_effects"]
                is_effects_enabled = state["effects_enabled"]

            processed_frame = frame
            # Apply the stack of filters in order, if enabled
            if is_effects_enabled:
                for effect_id in active_stack:
                    if effect_id in loaded_plugins:
                        processed_frame = loaded_plugins[effect_id].process(processed_frame)
            
            with lock:
                output_frame = processed_frame.copy()

            if args.debug:
                # Create a copy for display to avoid race conditions
                display_frame = processed_frame.copy()
                cv2.imshow('Processed Feed', display_frame)
                handle_debug_input()
            
    except (RuntimeError, BrokenPipeError) as e:
        logging.error(f"An error occurred: {e}")
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt detected. Exiting.")
    finally:
        running = False
        logging.info("Shutting down camera thread...")
        camera_thread.stop()
        camera_thread.join() # Wait for the thread to finish
        
        logging.info("Cleaning up resources...")
        cv2.destroyAllWindows()
        logging.info("Cleanup complete. Application terminated.")
        # The Uvicorn server thread is a daemon and will exit automatically
        sys.exit(0)

if __name__ == "__main__":
    main()
