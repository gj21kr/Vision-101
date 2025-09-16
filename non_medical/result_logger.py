"""Result logging utilities for non-medical modules."""
from medical.result_logger import ResultLogger


def create_logger_for_generating(algorithm_name: str, dataset_type: str = "") -> ResultLogger:
    """Create a logger for generating algorithms.

    Args:
        algorithm_name: Name of the generating algorithm.
        dataset_type: Optional dataset descriptor to append to the experiment name.

    Returns:
        A :class:`ResultLogger` configured for the generating workflow.
    """
    exp_name = f"generating_{algorithm_name}"
    if dataset_type:
        exp_name += f"_{dataset_type}"
    return ResultLogger(exp_name)


def create_logger_for_3d(algorithm_name: str, scene_type: str = "") -> ResultLogger:
    """Create a logger for 3D vision algorithms.

    Args:
        algorithm_name: Name of the 3D algorithm.
        scene_type: Optional descriptor for the scene being reconstructed.

    Returns:
        A :class:`ResultLogger` configured for 3D experiments.
    """
    exp_name = f"3d_{algorithm_name}"
    if scene_type:
        exp_name += f"_{scene_type}"
    return ResultLogger(exp_name)


__all__ = [
    "ResultLogger",
    "create_logger_for_generating",
    "create_logger_for_3d",
]
