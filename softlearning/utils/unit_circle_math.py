import numpy as np


def angle_distance_from_positions(point1, point2, keepdims=False):
    """Given two points on a unit circle, compute their angle distance."""
    angle1 = np.arctan2(*point1) + np.pi  # [0, 2pi]
    angle2 = np.arctan2(*point2) + np.pi  # [0, 2pi]
    distance = np.linalg.norm(
        angle1 - angle2, ord=1, keepdims=keepdims, axis=1)

    distance[distance > np.pi] = 2 * np.pi - distance[distance > np.pi]

    return distance


def position_from_angle(angle):
    """Return position (in format (y, x)) on a circle for a given angle."""
    position = (np.sin(angle), np.cos(angle))
    return position
