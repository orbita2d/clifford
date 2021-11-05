import numpy as np
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.cm
from .attractor import ArrayCounts


def get_intensity(k: float, arr: ArrayCounts) -> np.ndarray:
    return 1 - np.exp(- (k * arr.size()**2 / arr.count()) * arr.count_array)


class RenderSettings:
    """ Abstract base class for renderer. Renderers take clifford attractor data as density field and turn to rgb. """
    def __init__(self, data: dict):
        self.type: str = data["type"]
        self.data: dict = data["data"]

    def get_rgb(self, arr: ArrayCounts) -> np.ndarray:
        """ Return rgb from point density object."""
        raise SyntaxError('RenderSettings is an abstract base class. get_rgb() should be overwritten')


class LinearRenderer(RenderSettings):
    """ Linear renderer uses a matplotlib colourmap with (exponentially scaled) point density as input."""
    def __init__(self, data: dict):
        super(LinearRenderer, self).__init__(data)
        self.invert: bool = data["data"]["invert"]
        self.map = plt.get_cmap(self.data["map"], lut=self.data["lut"])
        self.alpha = self.data["alpha"]

    def get_rgb(self, arr: ArrayCounts):
        intensity = get_intensity(self.alpha, arr)
        if self.invert:
            value = 1 - intensity
        else:
            value = intensity
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        scalar_map = mpl.cm.ScalarMappable(norm=norm, cmap=self.map)
        rgb = scalar_map.to_rgba(value)
        return rgb


class DualLinearRenderer(RenderSettings):
    """ Dual linear renderer used two matplotlib colourmaps to colour the images. Some additional metric, like dx,
    dy, dr is used to interpolate between the two colours."""
    def __init__(self, data: dict):
        super(DualLinearRenderer, self).__init__(data)
        self.invert: bool = data["data"]["invert"]
        self.map1 = plt.get_cmap(self.data["map1"], lut=self.data["lut"])
        self.map2 = plt.get_cmap(self.data["map2"], lut=self.data["lut"])
        self.alpha = self.data["alpha"]
        # Can be "dx, dy, dr, dx2, dy2, dr2"
        self.select: str = self.data["select"]
        self.exponent = self.data["exponent"]

    def get_rgb_select(self, arr: ArrayCounts, select: np.ndarray):
        intensity = get_intensity(self.alpha, arr)
        if self.invert:
            value = 1 - intensity
        else:
            value = intensity
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        scalar_map1 = mpl.cm.ScalarMappable(norm=norm, cmap=self.map1)
        rgb1 = scalar_map1.to_rgba(value)
        scalar_map2 = mpl.cm.ScalarMappable(norm=norm, cmap=self.map2)
        rgb2 = scalar_map2.to_rgba(value)
        select_array = np.repeat(np.expand_dims(select, axis=2), 4, axis=2)
        rgb = rgb2 * select_array + rgb1 * (1-select_array)
        return rgb

    def get_rgb(self, arr: ArrayCounts):
        with np.errstate(divide='ignore', invalid='ignore'):
            # Normalise between [-1, 1]
            dx: np.ndarray = arr.dx / arr.count_array
            dy: np.ndarray = arr.dy / arr.count_array
            d2x: np.ndarray = arr.d2x / arr.count_array
            d2y: np.ndarray = arr.d2y / arr.count_array
            np.nan_to_num(dx, copy=False, nan=0.0, posinf=None, neginf=None)
            np.nan_to_num(dy, copy=False, nan=0.0, posinf=None, neginf=None)
            np.nan_to_num(d2x, copy=False, nan=0.0, posinf=None, neginf=None)
            np.nan_to_num(d2y, copy=False, nan=0.0, posinf=None, neginf=None)

        if self.select == 'dx':
            sel = (dx + 1) / 2
            return self.get_rgb_select(arr, np.power(sel, self.exponent))
        elif self.select == 'dy':
            sel = (dy + 1) / 2
            return self.get_rgb_select(arr, np.power(sel, self.exponent))
        elif self.select == 'dr':
            # dx in [-1, 1]
            # dy in [-1, 1]
            # dr between [0, sqrt(2)]
            sel = np.sqrt(dx ** 2 + dy ** 2) / 1.41421356237
            return self.get_rgb_select(arr, np.power(sel, self.exponent))
        elif self.select == 'dx2':
            sel = dx**2
            return self.get_rgb_select(arr, np.power(sel, self.exponent))
        elif self.select == 'dy2':
            sel = dy**2
            return self.get_rgb_select(arr, np.power(sel, self.exponent))
        elif self.select == 'dr2':
            sel = (dx ** 2 + dy ** 2) / 2
            return self.get_rgb_select(arr, np.power(sel, self.exponent))
        elif self.select == 'd2x':
            sel = (d2x + 1) / 2
            return self.get_rgb_select(arr, np.power(sel, self.exponent))
        elif self.select == 'd2y':
            sel = (d2x + 1) / 2
            return self.get_rgb_select(arr, np.power(sel, self.exponent))
        elif self.select == 'd2':
            sel = ((d2y + d2x) + 2) / 4
            return self.get_rgb_select(arr, np.power(sel, self.exponent))


class ColourSelectRenderer(RenderSettings):
    """ Use some metric (dx, dy, dr etc) to pick a colour from a mpl colourmap. Interpolate against the background
    with density value. """
    def __init__(self, data: dict):
        super(ColourSelectRenderer, self).__init__(data)
        self.invert: bool = data["data"]["invert"]
        self.map = plt.get_cmap(self.data["map"], lut=self.data["lut"])
        self.alpha = self.data["alpha"]

        # Can be "dx, dy, dr"
        self.select: str = data["data"]["select"]

    def get_rgb_hue(self, arr: ArrayCounts, hue_select: np.ndarray):
        intensity = get_intensity(self.alpha, arr)

        if self.invert:
            bg = np.array([1, 1, 1, 1])
        else:
            bg = np.array([0, 0, 0, 1])

        bg = np.reshape(bg, (1, 1, 4))
        bg_array = np.repeat(np.repeat(bg, arr.size(), axis=0), arr.size(), axis=1)
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
            sel = (dx + 1) / 2
            return self.get_rgb_hue(arr, sel)
        elif self.select == 'dy':
            # Renormalise between [0, 1]
            sel = (dy + 1) / 2
            return self.get_rgb_hue(arr, sel)
        elif self.select == 'dr':
            # dx in [-1, 1]
            # dy in [-1, 1]
            # dr between [0, sqrt(2)]
            sel = np.sqrt(dx**2 + dy**2) / 1.41421356237
            return self.get_rgb_hue(arr, sel)


class HSVRenderer(RenderSettings):
    """ Use some metric (dx, dy, dr etc) to pick hue. Use density to choose value and saturation."""
    def __init__(self, data: dict):
        super(HSVRenderer, self).__init__(data)
        self.invert: bool = data["data"]["invert"]
        # Can be "dx, dy, dr, dx2, dy2, dr2"
        self.select: str = data["data"]["select"]
        self.alpha: float = data["data"]["alpha"]
        self.beta: float = data["data"]["beta"]

    def get_rgb(self, arr: ArrayCounts):
        with np.errstate(divide='ignore', invalid='ignore'):
            # Normalise between [-1, 1]
            dx = arr.dx / arr.count_array / 2
            dy = arr.dy / arr.count_array / 2
            np.nan_to_num(dx, copy=False, nan=0.0, posinf=None, neginf=None)
            np.nan_to_num(dy, copy=False, nan=0.0, posinf=None, neginf=None)

        intensity = get_intensity(self.alpha, arr)
        sat = get_intensity(self.beta, arr)

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
    elif data["type"] == "bilinear":
        return DualLinearRenderer(data)
    else:
        raise AttributeError(f'Do not support renderer: {["type"]}')
