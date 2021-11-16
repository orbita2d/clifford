import numpy as np
from numba import jit
from typing import Tuple


@jit
def increment_clifford(x: float, y: float, a: float, b: float, c: float, d: float):
    """ Take a single point, increment it following the clifford attractor equations."""
    # Rescale so that attractor fits in [-1, 1]
    x_width = 1 + np.abs(c)
    y_width = 1 + np.abs(d)
    xn = (np.sin(a*y*y_width) + c * np.cos(a*x*x_width)) / x_width
    yn = (np.sin(b*x*x_width) + d * np.cos(b*y*y_width)) / y_width
    return xn, yn


@jit
def build_clifford(x: np.ndarray, y: np.ndarray, x0: float, y0: float, a: float, b: float, c: float, d: float, count: int):
    """Iterate through many points, producing array of points on attractor"""
    x[0], y[0] = increment_clifford(x0, y0, a, b, c, d)
    for i in range(1, count):
        x[i], y[i] = increment_clifford(x[i - 1], y[i - 1], a, b, c, d)


class ArrayCounts:
    """ Class for density array. Poitns are binned into pixels so we can make an image.
     Also track the change in x, and y for renderers."""
    def __init__(self, size: Tuple[int, int], padding: float):
        self.image_size: Tuple[int, int] = size
        self.real_size: int = min(size[0], size[1])
        self.count_array: np.ndarray = np.zeros(size, dtype=int)
        self.dx: np.ndarray = np.zeros(size, dtype=float)
        self.dy: np.ndarray = np.zeros(size, dtype=float)
        self.d2x: np.ndarray = np.zeros(size, dtype=float)
        self.d2y: np.ndarray = np.zeros(size, dtype=float)
        self.padding: float = padding

    def count(self):
        return self.count_array.sum()

    def size(self):
        return self.real_size


def update_counts(x0: float, y0: float, x: np.ndarray, y: np.ndarray, arr: ArrayCounts):
    """ Take array of points on attractor, append to density array with the size we've been given."""
    lpad: float = max(arr.image_size[0] - arr.image_size[1], 0) / 2 + arr.padding * arr.size()
    tpad: float = max(arr.image_size[1] - arr.image_size[0], 0) / 2 + arr.padding * arr.size()
    scale = arr.size() * (1 - 2 * arr.padding) / 2

    x_px = np.array((x + 1) * scale + lpad, dtype=int)
    y_px = np.array((y + 1) * scale + tpad, dtype=int)
    dx = np.diff(x, prepend=x0) / 2  # Dividing by two keeps dx in [-1, 1]
    dy = np.diff(y, prepend=y0) / 2
    np.add.at(arr.count_array, (x_px, y_px), 1)
    np.add.at(arr.dx, (x_px, y_px), dx)
    np.add.at(arr.dy, (x_px, y_px), dy)
    np.add.at(arr.d2x, (x_px, y_px), np.diff(dx, append=0) / 2)
    np.add.at(arr.d2y, (x_px, y_px), np.diff(dy, append=0) / 2)


def test_closed(p: np.ndarray):
    """ Check if a particular parameter is a small orbit which will make a bad image."""
    x0 = 0.08
    y0 = 0.12
    count = int(1E6)
    x = np.zeros(count, dtype=float)
    y = np.zeros(count, dtype=float)
    build_clifford(x, y, x0, y0, p[0], p[1], p[2], p[3], count)
    arr = ArrayCounts((320, 320), 0.0001)
    update_counts(x0, y0, x, y, arr)
    max_count = arr.count_array.max(initial=0)
    if (np.percentile(arr.count_array, 60) < 5) or (max_count > 10000):
        return True
    else:
        return False


def get_frame(x0: float, y0: float, p: np.ndarray, count: int, arr: ArrayCounts):
    """ Take some initial points in x, y and point in parameter space, increment the clifford atttractor count times
    and append them to the density array """
    x = np.zeros(count, dtype=float)
    y = np.zeros(count, dtype=float)
    build_clifford(x, y, x0, y0, p[0], p[1], p[2], p[3], count)
    update_counts(x0, y0, x, y, arr)
    return x[-1], y[-1]