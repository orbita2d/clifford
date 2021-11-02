import numpy as np
from numba import jit
from typing import Tuple


@jit
def increment_clifford(x: float, y: float, a: float, b: float, c: float, d: float):
    x_width = 1 + np.abs(c)
    y_width = 1 + np.abs(d)
    xn = (np.sin(a*y*y_width) + c * np.cos(a*x*x_width)) / x_width
    yn = (np.sin(b*x*x_width) + d * np.cos(b*y*y_width)) / y_width
    return xn, yn


@jit
def build_clifford(x: np.ndarray, y: np.ndarray, x0: float, y0: float, a: float, b: float, c: float, d: float, count: int):
    x[0], y[0] = increment_clifford(x0, y0, a, b, c, d)
    for i in range(1, count):
        x[i], y[i] = increment_clifford(x[i - 1], y[i - 1], a, b, c, d)


class ArrayCounts:
    def __init__(self, size: Tuple[int, int], padding: float):
        self.image_size: Tuple[int, int] = size
        self.real_size: int = min(size[0], size[1])
        self.count_array: np.ndarray = np.zeros(size, dtype=int)
        self.dx: np.ndarray = np.zeros(size, dtype=float)
        self.dy: np.ndarray = np.zeros(size, dtype=float)
        self.padding: float = padding

    def count(self):
        return self.count_array.sum()

    def size(self):
        return self.real_size


def update_counts(x0: float, y0: float, x: np.ndarray, y: np.ndarray, arr: ArrayCounts):
    lpad: float = max(arr.image_size[0] - arr.image_size[1], 0) / 2 + arr.padding * arr.size()
    tpad: float = max(arr.image_size[1] - arr.image_size[0], 0) / 2 + arr.padding * arr.size()
    scale = arr.size() * (1 - 2 * arr.padding) / 2

    x_px = np.array((x + 1) * scale + lpad, dtype=int)
    y_px = np.array((y + 1) * scale + tpad, dtype=int)
    dx = np.diff(x, prepend=x0)
    dy = np.diff(y, prepend=y0)
    np.add.at(arr.count_array, (x_px, y_px), 1)
    np.add.at(arr.dx, (x_px, y_px), dx)
    np.add.at(arr.dy, (x_px, y_px), dy)


def test_closed(p: np.ndarray):
    x0 = 0.08
    y0 = 0.12
    count = 400000
    x = np.zeros(count, dtype=float)
    y = np.zeros(count, dtype=float)
    build_clifford(x, y, x0, y0, p[0], p[1], p[2], p[3], count)
    arr = ArrayCounts((400, 400), 0.0005)
    update_counts(x0, y0, x, y, arr)
    nonzero = np.count_nonzero(arr.count_array)
    max_count = arr.count_array.max(initial=0)
    if (nonzero < 20000) or (max_count > 2000):
        return True
    else:
        return False


def get_frame(x0: float, y0: float, p: np.ndarray, count: int, arr: ArrayCounts):
    x = np.zeros(count, dtype=float)
    y = np.zeros(count, dtype=float)
    build_clifford(x, y, x0, y0, p[0], p[1], p[2], p[3], count)
    update_counts(x0, y0, x, y, arr)
    return x[-1], y[-1]