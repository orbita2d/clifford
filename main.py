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

mp_cores = 8

size = 800

blocks = 1
block_size = int(5E6)

padding = .04
alpha = 0.5

# Clifford attractor parameters
#[-1.974, -1.873, 1.0, -1.0]

p = np.array([1.6, -1.7, 1.5, -1.1], dtype=float)
delta_x = np.array([0.12, 0., 0.0, 0.], dtype=float)
delta_y = np.array([0., 0.01, 0., 0.1], dtype=float)

output_location = './frames/'
frames = 240

swap_luminance = False
use_saturation_map = True
beta = 0.15

# dx, dy, dr, dtheta, dvector, linear
colouring_schema = 'linear'
cmap = plt.get_cmap('inferno', lut=16)

alpha *= size / 6E3 * 1E6 / block_size
beta *= size / 4E3 * 1E6 / block_size


def update_image(arr: clifford.ArrayCounts, alpha: float, blocks_done: int) -> np.ndarray:
    intensity = 1 - np.exp(- alpha / blocks_done * arr.count_array)
    if use_saturation_map:
        sat = 1 - np.exp(- beta / blocks_done * arr.count_array)
    else:
        sat = np.ones(intensity.shape)
    if swap_luminance:
        value = 1 - intensity
    else:
        value = intensity
    with np.errstate(divide='ignore', invalid='ignore'):
        # Normalise between [-1, 1]
        dx = arr.dx / arr.count_array / 2
        dy = arr.dy / arr.count_array / 2
        np.nan_to_num(dx, copy=False, nan=0.0, posinf=None, neginf=None)
        np.nan_to_num(dy, copy=False, nan=0.0, posinf=None, neginf=None)

    if colouring_schema == 'dx':
        # Renormalise between [0, 1]
        dx = (dx + 1) / 2
        hue = dx
        hsv = np.dstack((hue, sat, value))
        rgb = mpl.colors.hsv_to_rgb(hsv)
        return rgb
    elif colouring_schema == 'dy':
        # Renormalise between [0, 1]
        dy = (dy + 1) / 2
        hue = dy
        hsv = np.dstack((hue, sat, value))
        rgb = mpl.colors.hsv_to_rgb(hsv)
        return rgb
    elif colouring_schema == 'dr':
        # Renormalise between [0, 1]
        dx = (dx + 1) / 2
        dy = (dy + 1) / 2
        hue = np.sqrt(dx**2 + dy**2) / 1.41421356237
        hsv = np.dstack((hue, sat, value))
        rgb = mpl.colors.hsv_to_rgb(hsv)
        return rgb
    elif colouring_schema == 'dtheta':
        # dtheta in [-pi, pi]
        dtheta = np.arctan2(dy, dx)
        hue = .5 + dtheta / (2 * np.pi)
        hsv = np.dstack((hue, sat, value))
        rgb = mpl.colors.hsv_to_rgb(hsv)
        return rgb
    elif colouring_schema == 'dvector':
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
    elif colouring_schema == 'linear':
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        scalar_map = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        rgb = scalar_map.to_rgba(value)
        return rgb


def gifsicle_cleanup(input: str, output: str, delay: Optional[int], colours: int):
    if delay is None:
        os.system(f'gifsicle -i -l0 --colors {colours} -O3 -j8 {input} -o {output}')
    else:
        os.system(f'gifsicle -i -l0 --colors {colours} -O3 -d {delay} -j8 {input} -o {output}')


def make_gifsicle(images_path: str, output: str):
    os.system(f'gifsicle -m {os.path.join(images_path, "clifford*.gif")} -o {output}')
    gifsicle_cleanup(output, output, 3, 16)


def make_gif(images_path: str, output: str, delay: float):
    os.system(f'convert -antialias -delay {delay*100:.5f} -loop 0 {os.path.join(images_path, "clifford*.gif")} {output}')


def solve_frame(fi: int, n: int, p0: np.ndarray, px: np.ndarray, py: np.ndarray):
    counts = clifford.ArrayCounts(size, padding)
    imdata = counts.count_array
    path_theta = 2 * np.pi * fi / n
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
                print(f'{fi} Found solutuion: delta = {delta} : p {p_test}')
                break
        if not found_solution:
            print(f'{fi} / {n} Failed')
            return
    t = time.perf_counter()
    for block in range(blocks):
        x0, y0 = clifford.get_frame(x0, y0, p_frame, block_size, counts)
        imdata = update_image(counts, alpha, block + 1)

    print(f'Done: {fi} / {n} frame in {time.perf_counter() - t:.3f}s')
    save_path = os.path.join(output_location, f'clifford{fi:04d}.gif')
    plt.imsave(save_path, imdata)


if __name__ == '__main__':
    frame_list = glob.glob(os.path.join(output_location, 'clifford*.gif'))
    for frame in frame_list:
        os.remove(frame)

    def solve_frame_p(x):
        solve_frame(x, frames, p, delta_x, delta_y)

    t = time.perf_counter()
    with multiprocessing.Pool(mp_cores) as pool:
        pool.imap(solve_frame_p, range(frames))
        pool.close()
        pool.join()
    time_used = time.perf_counter() - t
    print(f'Done: {frames} frames in {int(time_used/60):d} min {time_used % 60:.1f}s')
    print(f'Rendering gif')
    make_gif(output_location, os.path.join(output_location, f'out_convert.gif'), 1/30)
    gifsicle_cleanup(os.path.join(output_location, f'out_convert.gif'),
                     os.path.join(output_location, f'out_convert.gif'),
                     None, 16)
    make_gifsicle(output_location, os.path.join(output_location, f'out_gifsicle.gif'))
    print(f'Finished')
