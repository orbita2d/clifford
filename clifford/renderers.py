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
        self.map = plt.get_cmap(self.data["map"], lut=self.data["lut"])

        # Can be "dx, dy, dr"
        self.select: str = data["data"]["select"]

    def get_rgb_hue(self, arr: ArrayCounts, hue_select: np.ndarray):
        alpha: float = self.data["alpha"] * arr.size / 6E3 * 1E6 / arr.count()
        intensity: np.ndarray = 1 - np.exp(- alpha * arr.count_array)

        if self.invert:
            bg = np.array([1, 1, 1, 1])
        else:
            bg = np.array([0, 0, 0, 1])

        bg = np.reshape(bg, (1, 1, 4))
        bg_array = np.repeat(np.repeat(bg, arr.size, axis=0), arr.size, axis=1)
        intensity_array = np.repeat(np.expand_dims(intensity, axis=2), 4, axis=2)
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        scalar_map = mpl.cm.ScalarMappable(norm=norm, cmap=self.map)
        colours: np.ndarray = scalar_map.to_rgba(hue_select)

        rgb = bg_array * (1 - intensity_array) + colours * intensity_array
        return rgb

    def get_rgb(self, arr: ArrayCounts):
        with np.errstate(divide='ignore', invalid='ignore'):
            # Normalise between [-1, 1]
            dx = arr.dx / arr.count_array / 2
            dy = arr.dy / arr.count_array / 2
            np.nan_to_num(dx, copy=False, nan=0.0, posinf=None, neginf=None)
            np.nan_to_num(dy, copy=False, nan=0.0, posinf=None, neginf=None)

        if self.select == 'dx':
            # Renormalise between [0, 1]
            dx = (dx + 1) / 2
            return self.get_rgb_hue(arr, dx)
        elif self.select == 'dy':
            # Renormalise between [0, 1]
            dy = (dy + 1) / 2
            return self.get_rgb_hue(arr, dy)
        elif self.select == 'dr':
            # Renormalise between [0, 1]
            dx = (dx + 1) / 2
            dy = (dy + 1) / 2
            dr = np.sqrt(dx**2 + dy**2) / 1.41421356237
            return self.get_rgb_hue(arr, dr)


class HSVRenderer(RenderSettings):
    def __init__(self, data: dict):
        super(HSVRenderer, self).__init__(data)
        self.invert: bool = data["data"]["invert"]
        # Can be "dx, dy, dr, dx2, dy2, dr2"
        self.select: str = data["data"]["select"]
        self.alpha: float = data["data"]["alpha"]
        self.beta: float = data["data"]["beta"]

    def get_rgb(self, arr: ArrayCounts):
        alpha: float = self.alpha * arr.size / 6E3 * 1E6 / arr.count()
        beta: float = self.beta * arr.size / 6E3 * 1E6 / arr.count()
        with np.errstate(divide='ignore', invalid='ignore'):
            # Normalise between [-1, 1]
            dx = arr.dx / arr.count_array / 2
            dy = arr.dy / arr.count_array / 2
            np.nan_to_num(dx, copy=False, nan=0.0, posinf=None, neginf=None)
            np.nan_to_num(dy, copy=False, nan=0.0, posinf=None, neginf=None)

        intensity = 1 - np.exp(- alpha * arr.count_array)
        sat = 1 - np.exp(- beta * arr.count_array)

        if self.invert:
            value = 1 - intensity
        else:
            value = intensity

        if self.select == 'dx':
            # Renormalise between [0, 1]
            hue = (dx + 1) / 2
            hsv = np.dstack((hue, sat, value))
            rgb = mpl.colors.hsv_to_rgb(hsv)
            return rgb
        elif self.select == 'dy':
            # Renormalise between [0, 1]
            hue = (dy + 1) / 2
            hsv = np.dstack((hue, sat, value))
            rgb = mpl.colors.hsv_to_rgb(hsv)
            return rgb
        elif self.select == 'dr':
            # dx in [-1, 1]
            # dy in [-1, 1]
            # dr between [0, sqrt(2)]
            hue = np.sqrt(dx ** 2 + dy ** 2) / 1.41421356237
            hsv = np.dstack((hue, sat, value))
            rgb = mpl.colors.hsv_to_rgb(hsv)
            return rgb
        if self.select == 'dx2':
            # dx in [-1, 1]
            hue = dx**2
            hsv = np.dstack((hue, sat, value))
            rgb = mpl.colors.hsv_to_rgb(hsv)
            return rgb
        elif self.select == 'dy2':
            # dy in [-1, 1]
            hue = dy**2
            hsv = np.dstack((hue, sat, value))
            rgb = mpl.colors.hsv_to_rgb(hsv)
            return rgb
        elif self.select == 'dr2':
            # dx in [-1, 1]
            # dy in [-1, 1]
            # dr between [0, sqrt(2)]
            hue = (dx ** 2 + dy ** 2) / 2
            hsv = np.dstack((hue, sat, value))
            rgb = mpl.colors.hsv_to_rgb(hsv)
            return rgb


def get_renderer(data: dict) -> RenderSettings:
    if data["type"] == "linear":
        return LinearRenderer(data)
    elif data["type"] == "colour-select":
        return ColourSelectRenderer(data)
    elif data["type"] == "hsv":
        return HSVRenderer(data)
    else:
        raise AttributeError(f'Do not support renderer: {["type"]}')
