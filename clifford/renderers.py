import numpy as np
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.cm
from .attractor import ArrayCounts


class RenderSettings:
    def __init__(self, data: dict):
        self.type: str = data["type"]
        self.data: dict = data["data"]

    def get_rgb(self, arr: ArrayCounts):
        raise SyntaxError('RenderSettings is an abstract base class. get_rgb() should be overwritten')


class LinearRenderer(RenderSettings):
    def __init__(self, data: dict):
        super(LinearRenderer, self).__init__(data)
        self.invert: bool = data["data"]["invert"]
        self.map = plt.get_cmap(self.data["map"], lut=self.data["lut"])

    def get_rgb(self, arr: ArrayCounts):
        alpha: float = self.data["alpha"] * arr.size / 6E3 * 1E6 / arr.count()
        intensity = 1 - np.exp(- alpha * arr.count_array)
        if self.invert:
            value = 1 - intensity
        else:
            value = intensity
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        scalar_map = mpl.cm.ScalarMappable(norm=norm, cmap=self.map)
        rgb = scalar_map.to_rgba(value)
        return rgb


class ColourSelectRenderer(RenderSettings):
    def __init__(self, data: dict):
        super(ColourSelectRenderer, self).__init__(data)
        self.invert: bool = data["data"]["invert"]

    def get_rgb_hue(self, arr: ArrayCounts, hue_select: np.ndarray):
        alpha: float = self.data["alpha"] * arr.size / 6E3 * 1E6 / arr.count()
        beta: float = self.data["beta"] * arr.size / 4E3 * 1E6 / arr.count()
        intensity = 1 - np.exp(- alpha * arr.count_array)

        sat = 1 - np.exp(- beta * arr.count_array)

        if self.invert:
            value = 1 - intensity
        else:
            value = intensity

        hsv = np.dstack((hue_select, sat, value))
        rgb = mpl.colors.hsv_to_rgb(hsv)
        return rgb

    def get_rgb(self, arr: ArrayCounts):
        with np.errstate(divide='ignore', invalid='ignore'):
            # Normalise between [-1, 1]
            dx = arr.dx / arr.count_array / 2
            dy = arr.dy / arr.count_array / 2
            np.nan_to_num(dx, copy=False, nan=0.0, posinf=None, neginf=None)
            np.nan_to_num(dy, copy=False, nan=0.0, posinf=None, neginf=None)

        if self.type == 'dx':
            # Renormalise between [0, 1]
            dx = (dx + 1) / 2
            return self.get_rgb_hue(arr, dx)
        elif self.type == 'dy':
            # Renormalise between [0, 1]
            dy = (dy + 1) / 2
            return self.get_rgb_hue(arr, dy)
        elif self.type == 'dr':
            # Renormalise between [0, 1]
            dx = (dx + 1) / 2
            dy = (dy + 1) / 2
            dr = np.sqrt(dx**2 + dy**2) / 1.41421356237
            return self.get_rgb_hue(arr, dr)


def get_renderer(data: dict) -> RenderSettings:
    if data["type"] == "linear":
        return LinearRenderer(data)
    elif data["type"] == "dx":
        return ColourSelectRenderer(data)
    elif data["type"] == "dy":
        return ColourSelectRenderer(data)
    elif data["type"] == "dr":
        return ColourSelectRenderer(data)
    else:
        raise AttributeError(f'Do not support renderer: {["type"]}')
