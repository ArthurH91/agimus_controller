import numpy as np
from dataclasses import dataclass


@dataclass
class MPCOutputData:
    """Output data structure of the MPC."""
    x: list  # List of numpy.ndarray, the output state.
    u: list  # List of numpy.ndarray, the output control.
    ricatti_gains: list  # List of numpy.ndarray, the Ricatti gains.
    feed_forward_terms: list  # List of numpy.ndarray, the feed forward terms.


@dataclass
class MPCDebugData:
    # Solver infos
    problem_solved: bool
    others: int  # TODO: TBD

    # Debug data
    list_robot_configuration: list
    list_robot_velocity: list
    list_robot_acceleration: list
    list_robot_effort: list
    list_forces: list
    list_end_effector_pose: list
