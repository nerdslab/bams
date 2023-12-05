import numpy as np


def diff(vec, axis=-1, h=1, padding="edge"):
    assert padding in [
        "zero",
        "edge",
    ], "Padding must be one of ['zero', 'edge'],"
    " got {}.".format(padding)

    # move the target axis to the end
    vec = np.moveaxis(vec, axis, -1)

    # compute diff
    dvec = np.zeros_like(vec)
    dvec[..., h:] = vec[..., h:] - vec[..., :-h]

    # take care of padding the beginning
    if padding == "edge":
        for i in range(h):
            dvec[..., i] = dvec[..., h + 1]

    # move the axis back to its original position
    dvec = np.moveaxis(dvec, -1, axis)
    return dvec


def to_polar_coordinates(vec):
    r = np.linalg.norm(vec, axis=-1)
    theta = np.arctan2(vec[..., 1], vec[..., 0])
    return r, theta


def to_cartasian_coordinates(r, theta):
    x, y = r * np.cos(theta), r * np.sin(theta)
    return x, y


def angle_clip(theta):
    return np.mod(theta + np.pi, 2 * np.pi) - np.pi

