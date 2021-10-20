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

mp_cores = 12

size = 1000

blocks = 10
block_size = int(5E6)

padding = .04
alpha = 0.5

# Clifford attractor parameters
#[-1.974, -1.873, 1.0, -1.0]

p = np.array([1.8124, -1.973, 1.3, -1.0], dtype=float)
delta_x = np.array([0.02, 0., 0.1, 0.], dtype=float)
delta_y = np.array([0., 0.01, 0., 0.1], dtype=float)

output_location = './frames/'
frames = 300

swap_luminance = False
use_saturation_map = True
beta = 0.15

# dx, dy, dr, dtheta, dvector, linear
colouring_schema = 'linear'
cmap = plt.get_cmap('magma')

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


def make_gif(images_path: str, output: str, delay: float):
    cmd = 'convert' + ' -delay ' + str(delay*100) + ' -loop 0 ' + os.path.join(images_path, '*.jpeg') + ' ' + output
    os.system(cmd)


def solve_frame(fi: int, n: int, p: np.ndarray):
    counts = clifford.ArrayCounts(size, padding)
    imdata = counts.count_array
    path_theta = 2 * np.pi * fi / n
    p_frame = p + np.cos(path_theta) * delta_x + np.sin(path_theta) * delta_y
    x0 = 0.02
    y0 = 0.12
    if clifford.test_closed(p_frame):
        return
    t = time.perf_counter()
    for block in range(blocks):
        x0, y0 = clifford.get_frame(x0, y0, p_frame, block_size, counts)
        imdata = update_image(counts, alpha, block + 1)
    print(f'Done: {fi} / {n} frame in {time.perf_counter() - t:.3f}s')
    save_path = os.path.join(output_location, f'clifford{fi:04d}.jpeg')
    plt.imsave(save_path, imdata)


if __name__ == '__main__':
    frame_list = glob.glob(os.path.join(output_location, 'clifford*.jpeg'))
    for frame in frame_list:
        os.remove(frame)

    def solve_frame_p(x):
        solve_frame(x, frames, p)

    t = time.perf_counter()
#    for i in range(frames):
#        solve_frame_p(i)
    with multiprocessing.Pool(mp_cores) as pool:
        pool.imap(solve_frame_p, range(frames))
        pool.close()
        pool.join()
    print(f'Done: {frames} frames in {time.perf_counter() - t:.1f}s')
    print(f'Rendering gif')
    make_gif(output_location, os.path.join(output_location, f'clifford.gif'), 1/60)
    print(f'Finished')
