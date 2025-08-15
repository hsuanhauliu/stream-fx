from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, List, Optional

class BaseFilter(ABC):
    """
    Abstract base class for all filter plugins.
    Each filter must implement these properties and methods.
    """

    @property
    @abstractmethod
    def identifier(self) -> str:
        """A unique, machine-readable identifier for the filter (e.g., 'sepia_tone')."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """A human-readable name for the filter (e.g., 'Sepia Tone')."""
        pass

    @property
    def category(self) -> str:
        """
        A string defining the category for UI grouping (e.g., 'Artistic', 'Utility').
        If not implemented, it will default to a generic category.
        """
        return "General"

    def get_parameters(self) -> List[Dict[str, Any]]:
        """
        Returns a list of dictionaries, each defining a configurable parameter.
        This allows the UI to dynamically generate controls.
        """
        return []

    def update_parameters(self, params: Dict[str, Any]):
        """
        Receives a dictionary of updated parameter values from the UI.
        The filter is responsible for updating its internal state.
        """
        pass

    def initialize(self, config: Dict[str, Any] = None):
        """
        Optional method to initialize resources when the application starts.
        This is useful for loading models or setting up detectors.
        """
        pass
    
    def on_deactivate(self):
        """
        Called when the filter is removed from the active stack or disabled.
        Useful for cleaning up resources like network connections.
        """
        pass

    @abstractmethod
    def process(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Process a single video frame and return the modified frame.
        If the filter is temporary and its effect is finished, it can
        return None to signal its removal from the active stack.
        """
        pass
