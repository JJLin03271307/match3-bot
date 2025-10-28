from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class ScreenProvider(ABC):
    @abstractmethod
    def grab(self) -> np.ndarray:
        """Capture full monitor frame (BGR)."""
        ...

    @abstractmethod
    def grab_roi(self, roi: Tuple[int, int, int, int]) -> np.ndarray:
        """Capture ROI relative to the monitor (BGR)."""
        ...


class InputController(ABC):
    @abstractmethod
    def drag(self, x1: int, y1: int, x2: int, y2: int, ms: int = 150): ...

    @abstractmethod
    def click(self, x: int, y: int): ...
