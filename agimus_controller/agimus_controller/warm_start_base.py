from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from agimus_controller.ocp_base import OCPBase
from agimus_controller.trajectory import TrajectoryPoint


class WarmStartBase(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def setup(self, ocp: OCPBase = None) -> None:
        pass

    @abstractmethod
    def generate(self, reference_trajectory: list[TrajectoryPoint]) -> tuple(
        np.ndarray, np.ndarray
    ):
        """Returns x_init, u_init."""
        pass
