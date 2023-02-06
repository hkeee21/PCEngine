from pyquaternion import Quaternion
import numpy as np


def transformation_matrix_from_quaternion(translation, rotation):
    x, y, z = translation
    q = Quaternion(rotation)
    m = np.eye(4)
    rot = q.rotation_matrix
    m[:3, :3] = rot
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def to_transformation_matrix(x, y, z, ry):
    q = Quaternion(axis=[0, 0, 1], radians=ry)
    m = np.eye(4)
    m[:3, :3] = q.rotation_matrix
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def from_transformation_matrix(m):
    x = m[0, 3]
    y = m[1, 3]
    z = m[2, 3]
    rotation = m[:3, :3]
    q = Quaternion(matrix=rotation)
    yaw = yaw_from_quaternion(q.q)
    return x, y, z, yaw


def transform(points, transf_matrix: np.ndarray) -> None:
    """
    Applies a homogeneous transform.
    :param transf_matrix: <np.float: 4, 4>. Homogenous transformation matrix.
    """
    # Adapted from nuscenes devkit. They use points in the transposed form.
    points = points.T
    points[:3, :] = transf_matrix.dot(
        np.vstack((points[:3, :], np.ones(points.shape[1]))))[:3, :]
    return points.T


def yaw_from_quaternion(quaternion):
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """
    q = Quaternion(quaternion)
    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw
