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
import yaml
from fastapi import FastAPI
from starlette.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os
import importlib
import inspect
from queue import Queue
from collections import defaultdict

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
def load_plugins_from_directory(directory: str, module_prefix: str, config: Dict[str, Any]) -> Dict[str, BaseFilter]:
    """Helper function to load plugins from a specific directory."""
    plugins = {}
    if not os.path.isdir(directory):
        logging.warning(f"Plugin directory not found: {directory}")
        return plugins
        
    for filename in os.listdir(directory):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = f"{module_prefix}.{filename[:-3]}"
            try:
                module = importlib.import_module(module_name)
                for item_name in dir(module):
                    item = getattr(module, item_name)
                    # Check if the item is a class, a subclass of BaseFilter, and not an abstract class
                    if isinstance(item, type) and issubclass(item, BaseFilter) and not inspect.isabstract(item):
                        plugin_instance = item()
                        # Pass the specific config for this plugin to its initializer
                        plugin_config = config.get('filters', {}).get(plugin_instance.identifier)
                        plugin_instance.initialize(plugin_config)
                        plugins[plugin_instance.identifier] = plugin_instance
                        logging.info(f"Successfully loaded plugin: '{plugin_instance.name}' ({plugin_instance.identifier})")
            except (ImportError, ModuleNotFoundError) as e:
                logging.warning(f"Could not load plugin from {filename} due to a missing dependency: {e}")
            except Exception as e:
                logging.error(f"An unexpected error occurred while loading plugin from {filename}: {e}")
    return plugins

def load_all_plugins(enable_advanced: bool, config: Dict[str, Any]) -> Dict[str, BaseFilter]:
    """Dynamically loads all filter plugins."""
    all_plugins = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load standard filters
    filters_dir_path = os.path.join(script_dir, "filters")
    all_plugins.update(load_plugins_from_directory(filters_dir_path, "filters", config))

    # Conditionally load advanced filters
    if enable_advanced:
        logging.info("Advanced filters enabled. Loading...")
        advanced_filters_dir_path = os.path.join(filters_dir_path, "advanced")
        all_plugins.update(load_plugins_from_directory(advanced_filters_dir_path, "filters.advanced", config))
    else:
        logging.info("Advanced filters disabled. To enable, use the --enable_advanced_filters flag.")
        
    return all_plugins

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads and parses the YAML configuration file."""
    if config_path:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logging.info(f"Successfully loaded configuration from {config_path}")
                return config if config is not None else {}
        except FileNotFoundError:
            logging.error(f"Configuration file not found at {config_path}. Using defaults.")
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML configuration file: {e}. Using defaults.")
    return {}


# --- FastAPI App and Global variables ---
app = FastAPI()
output_frame = None
lock = threading.Lock()

# --- Global State Management ---
state = {
    "active_effects": [], # A list of effect identifiers, defining the stack
    "effects_enabled": True, # Master toggle for the entire stack
    "parameter_values": {} # Stores current values for filter parameters
}
state_lock = threading.Lock()

# --- API Models ---
class ParameterInfo(BaseModel):
    identifier: str
    name: str
    type: str
    min: float | None = None
    max: float | None = None
    step: float | None = None
    default: Any
    options: List[str] | None = None # For select type

class FilterInfo(BaseModel):
    identifier: str
    name: str
    category: str
    parameters: List[ParameterInfo]

class ControlStatus(BaseModel):
    active_effects: List[str]
    available_filters_by_category: Dict[str, List[FilterInfo]]
    effects_enabled: bool
    parameter_values: Dict[str, Dict[str, Any]]

class SetStackRequest(BaseModel):
    effects: List[str] = Field(..., description="An ordered list of effect identifiers.")

class UpdateParameterRequest(BaseModel):
    filter_id: str
    params: Dict[str, Any]


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
        active_effects = list(state["active_effects"])
        effects_enabled = state["effects_enabled"]
        param_values = dict(state["parameter_values"])
    
    # Group available filters by category
    available_by_category = defaultdict(list)
    for plugin in loaded_plugins.values():
        info = {
            "identifier": plugin.identifier, 
            "name": plugin.name, 
            "category": plugin.category,
            "parameters": plugin.get_parameters()
        }
        available_by_category[plugin.category].append(info)

    return {
        "active_effects": active_effects, 
        "available_filters_by_category": available_by_category, 
        "effects_enabled": effects_enabled,
        "parameter_values": param_values
    }

@app.post("/control/set_stack", response_model=Dict)
async def set_stack(request: SetStackRequest):
    """Sets the active filter stack to a new configuration."""
    with state_lock:
        old_stack = set(state["active_effects"])
        new_stack_list = [eff for eff in request.effects if eff in loaded_plugins]
        new_stack = set(new_stack_list)

        # Call on_deactivate for filters that were removed from the stack
        removed_filters = old_stack - new_stack
        for filter_id in removed_filters:
            if filter_id in loaded_plugins:
                loaded_plugins[filter_id].on_deactivate()

        state["active_effects"] = new_stack_list
        logging.info(f"Filter stack updated via API: {new_stack_list}")
        return {"status": "success", "active_effects": new_stack_list}

@app.post("/control/toggle_all", response_model=Dict)
async def toggle_all_effects():
    """Toggles the entire filter stack on or off (disable)."""
    with state_lock:
        state["effects_enabled"] = not state["effects_enabled"]
        is_enabled = state["effects_enabled"]
        logging.info(f"All effects toggled via API. Enabled: {is_enabled}")

        # If effects are being disabled, call on_deactivate for all active filters
        if not is_enabled:
            for effect_id in state["active_effects"]:
                if effect_id in loaded_plugins:
                    loaded_plugins[effect_id].on_deactivate()

        return {"status": "success", "effects_enabled": is_enabled}

@app.post("/control/update_parameter", response_model=Dict)
async def update_parameter(request: UpdateParameterRequest):
    """Updates a specific parameter for a filter."""
    with state_lock:
        if request.filter_id in loaded_plugins:
            plugin = loaded_plugins[request.filter_id]
            plugin.update_parameters(request.params)
            
            # Update the central state
            if request.filter_id not in state["parameter_values"]:
                state["parameter_values"][request.filter_id] = {}
            state["parameter_values"][request.filter_id].update(request.params)

            logging.info(f"Updated parameters for '{request.filter_id}': {request.params}")
            return {"status": "success"}
        return {"status": "error", "message": "Filter not found"}


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
loaded_plugins = {}

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
    parser.add_argument('--port', type=int, default=None, help='Port for the streaming server.')
    parser.add_argument(
        '--camera',
        type=int,
        default=None,
        help='Index of the camera to use (e.g., 0, 1, 2).'
    )
    parser.add_argument(
        '--enable_advanced_filters',
        action='store_true',
        help='Enable loading of advanced filters from the filters/advanced directory.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to the YAML configuration file.'
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
    global output_frame, lock, running, state, state_lock, loaded_plugins
    
    args = parse_arguments()
    
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load configuration and then plugins
    config = load_config(args.config)
    loaded_plugins = load_all_plugins(args.enable_advanced_filters, config)

    # Initialize parameter values in the global state
    with state_lock:
        for plugin_id, plugin in loaded_plugins.items():
            state["parameter_values"][plugin_id] = {}
            for param in plugin.get_parameters():
                state["parameter_values"][plugin_id][param["identifier"]] = param["default"]


    # Determine final settings with precedence: CLI > config > default
    app_config = config.get('app', {})
    host = args.host if args.host is not None else app_config.get('host', '127.0.0.1')
    port = args.port if args.port is not None else app_config.get('port', 8080)
    camera_index = args.camera if args.camera is not None else app_config.get('camera', 0)

    signal.signal(signal.SIGINT, signal_handler)
    
    # --- FastAPI Server Thread ---
    uvicorn_config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(uvicorn_config)
    server_thread = threading.Thread(target=server.run)
    server_thread.daemon = True
    server_thread.start()
    logging.info(f"Streaming server started at http://{host}:{port}/video_feed")
    logging.info(f"Control UI available at http://{host}:{port}/")

    # --- Camera and Processing Thread ---
    # A queue of size 1 will always hold the latest frame
    frame_queue = Queue(maxsize=1)
    camera_thread = CameraThread(camera_index, frame_queue)
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
                active_stack = list(state["active_effects"])
                is_effects_enabled = state["effects_enabled"]

            processed_frame = frame
            effects_to_remove = []

            # Apply the stack of filters in order, if enabled
            if is_effects_enabled:
                for effect_id in active_stack:
                    if effect_id in loaded_plugins:
                        result_frame = loaded_plugins[effect_id].process(processed_frame)
                        
                        if result_frame is None:
                            # The filter has signaled that it's finished
                            effects_to_remove.append(effect_id)
                        else:
                            processed_frame = result_frame
            
            # If any filters need to be removed, update the state
            if effects_to_remove:
                with state_lock:
                    state["active_effects"] = [eff for eff in state["active_effects"] if eff not in effects_to_remove]
                logging.info(f"Auto-removed finished filters: {effects_to_remove}")

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
