from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any


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

    def initialize(self, config: Dict[str, Any] = None):
        """
        Optional method to initialize resources when the application starts.
        This is useful for loading models or setting up detectors.
        By default, it does nothing.
        """
        pass

    @abstractmethod
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single video frame and return the modified frame.
        This method must be implemented by all plugins.
        """
        pass
