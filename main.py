import matplotlib as mpl
import matplotlib.colors
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import time
import clifford
import os.path
import glob
import multiprocessing
from typing import Optional
import json
import sys

block_size = int(5E6)

padding = .02

output_location = './frames/'


class RenderSettings:
    def __init__(self, data: dict):
        self.type: str = data["type"]
        self.saturation_map: bool = data["data"]["saturation_map"]
        self.invert: bool = data["data"]["invert"]
        self.data: dict = data["data"]


class GeneratorSettings:
    def __init__(self, data: dict):
        self.p0: np.ndarray = np.array(data["p0"])
        self.px: np.ndarray = np.array(data["px"])
        self.py: np.ndarray = np.array(data["py"])


def update_image(arr: clifford.ArrayCounts, blocks_done: int, settings: RenderSettings) -> np.ndarray:
    alpha: float = settings.data["alpha"] * arr.size / 6E3 * 1E6 / block_size
    beta: float = settings.data["beta"] * arr.size / 4E3 * 1E6 / block_size

    intensity = 1 - np.exp(- alpha / blocks_done * arr.count_array)
    if settings.saturation_map:
        sat = 1 - np.exp(- beta / blocks_done * arr.count_array)
    else:
        sat = np.ones(intensity.shape)
    if settings.invert:
        value = 1 - intensity
    else:
        value = intensity
    with np.errstate(divide='ignore', invalid='ignore'):
        # Normalise between [-1, 1]
        dx = arr.dx / arr.count_array / 2
        dy = arr.dy / arr.count_array / 2
        np.nan_to_num(dx, copy=False, nan=0.0, posinf=None, neginf=None)
        np.nan_to_num(dy, copy=False, nan=0.0, posinf=None, neginf=None)

    if settings.type == 'dx':
        # Renormalise between [0, 1]
        dx = (dx + 1) / 2
        hue = dx
        hsv = np.dstack((hue, sat, value))
        rgb = mpl.colors.hsv_to_rgb(hsv)
        return rgb
    elif settings.type == 'dy':
        # Renormalise between [0, 1]
        dy = (dy + 1) / 2
        hue = dy
        hsv = np.dstack((hue, sat, value))
        rgb = mpl.colors.hsv_to_rgb(hsv)
        return rgb
    elif settings.type == 'dr':
        # Renormalise between [0, 1]
        dx = (dx + 1) / 2
        dy = (dy + 1) / 2
        hue = np.sqrt(dx**2 + dy**2) / 1.41421356237
        hsv = np.dstack((hue, sat, value))
        rgb = mpl.colors.hsv_to_rgb(hsv)
        return rgb
    elif settings.type == 'dtheta':
        # dtheta in [-pi, pi]
        dtheta = np.arctan2(dy, dx)
        hue = .5 + dtheta / (2 * np.pi)
        hsv = np.dstack((hue, sat, value))
        rgb = mpl.colors.hsv_to_rgb(hsv)
        return rgb
    elif settings.type == 'dvector':
        # dtheta in [-pi, pi]
        dtheta = np.arctan2(dy, dx)
        # Renormalise between [0, 1]
        dx = (dx + 1) / 2
        dy = (dy + 1) / 2
        dr = np.sqrt(dx**2 + dy**2) / 1.41421356237
        hue = .5 + dtheta / (2 * np.pi)
        sat *= dr
        hsv = np.dstack((hue, sat, value))
        rgb = mpl.colors.hsv_to_rgb(hsv)
        return rgb
    elif settings.type == 'linear':
        cmap = plt.get_cmap(settings.data["map"], lut=settings.data["lut"])
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        scalar_map = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        rgb = scalar_map.to_rgba(value)
        return rgb


def gifsicle_cleanup(input: str, output: str):
    os.system(f'gifsicle -i -l0 -d2 -O3 -j8 {input} -o {output}')


def make_gifsicle(images_path: str, output: str):
    os.system(f'gifsicle -m {os.path.join(images_path, "frame*.gif")} -o {output}')


def make_gif(images_path: str, output: str, delay: float):
    os.system(f'convert -antialias -delay {delay*100:.5f} -loop 0 {os.path.join(images_path, "frame*.png")} {output}')


def solve_frame(fi: int, blocks: int, n: int, generator: GeneratorSettings, settings: RenderSettings, size: int):
    counts = clifford.ArrayCounts(size, padding)
    path_theta = 2 * np.pi * fi / n
    p0 = generator.p0
    px = generator.px
    py = generator.py
    p_frame = p0 + np.cos(path_theta) * px + np.sin(path_theta) * py
    x0 = 0.08
    y0 = 0.12
    if clifford.test_closed(p_frame):
        print(f'{fi} / {n} - closed')
        p_delta = p0 - p_frame
        max_i = 100
        max_delta = 0.02
        found_solution = False
        for i in range(1, max_i+1):
            delta = i * max_delta / max_i
            p_test = p_frame + delta * p_delta
            if not clifford.test_closed(p_test):
                found_solution = True
                p_frame = p_test
                print(f'{fi} Found solutuion: delta = {delta:.3f}')
                break
        if not found_solution:
            print(f'{fi} / {n} Failed')
            return
    t = time.perf_counter()
    for block in range(blocks):
        x0, y0 = clifford.get_frame(x0, y0, p_frame, block_size, counts)

    imdata = update_image(counts, blocks, settings)
    print(f'Done: {fi} / {n} frame in {time.perf_counter() - t:.3f}s')
    plt.imsave(os.path.join(output_location, f'frame{fi:04d}.gif'), imdata)
    plt.imsave(os.path.join(output_location, f'frame{fi:04d}.png'), imdata)


if __name__ == '__main__':
    if len(sys.argv) <= 5:
        print('config blocks frames size cores')
        exit(1)
    config_path = sys.argv[1]
    blocks = int(sys.argv[2])
    frames = int(sys.argv[3])
    size = int(sys.argv[4])
    cores = int(sys.argv[5])
    with open(config_path, 'r') as f:
        json_data: dict = json.load(f)
        name = json_data["name"]
        if not json_data["type"] == "clifford":
            exit(1)
        generator_settings = GeneratorSettings(json_data["data"]["generation"])
        render_settings = RenderSettings(json_data["data"]["render"])

    print(f'Building {name}.gif')
    frame_list = glob.glob(os.path.join(output_location, 'frame*.gif'))
    for frame in frame_list:
        os.remove(frame)
    frame_list = glob.glob(os.path.join(output_location, 'frame*.png'))
    for frame in frame_list:
        os.remove(frame)

    def solve_frame_p(x):
        solve_frame(x, blocks, frames, generator_settings, render_settings, size)

    t = time.perf_counter()
    with multiprocessing.Pool(cores) as pool:
        pool.imap(solve_frame_p, range(frames))
        pool.close()
        pool.join()
    time_used = time.perf_counter() - t
    print(f'Done: {frames} frames in {int(time_used/60):d} min {time_used % 60:.1f}s')
    print(f'Rendering gif')
    convert_out = os.path.join(output_location, f'{name}.gif')
    convert_out2 = os.path.join(output_location, f'{name}.jic.gif')
    make_gif(output_location, convert_out2, 0.02)
    make_gifsicle(output_location, convert_out)
    gifsicle_cleanup(convert_out,
                     convert_out)
    print(f'Finished')
