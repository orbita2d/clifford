import matplotlib.pyplot as plt
import time
import clifford
import os.path
import glob
import multiprocessing
import json
import sys
from typing import Tuple

block_size = int(5E6)

output_location = './frames/'
debug = False


def gifsicle_cleanup(input: str, output: str):
    """ Make the gif have the right framerate, loop, etc"""
    os.system(f'gifsicle -i -l0 -d2 -O3 -j8 --colors 256 {input} -o {output}')


def make_gifsicle(images_path: str, output: str):
    """ Make a gif."""
    os.system(f'gifsicle --colors 256 -m {os.path.join(images_path, "frame*.gif")} -o {output}')


def make_mp4(images_path: str, output: str):
    """ Make an mp4 from <images_path>/frame%0d.png frames. Save to <output>"""
    cmd = f'ffmpeg -y -loglevel 0 -r 50 -f image2 -i {os.path.join(images_path, "frame%04d.png")} -vcodec libx264 ' \
          f'-crf 15  -pix_fmt yuv420p {output} '
    os.system(cmd)


def solve_frame(fi: int, blocks: int, n: int, generator: clifford.GeneratorSettings, renderer: clifford.RenderSettings,
                size: Tuple[int, int]):
    """ Take a frame count, number of iterations. Calculate clifford attractor point density, render as image,
    save frame. """
    counts = clifford.ArrayCounts(size, 0.05)
    x0 = 0.08
    y0 = 0.12
    p_frame = generator.get_p(fi, n, 0.0)
    if clifford.test_closed(p_frame):
        print(f'{fi} / {n} - closed')
        max_i = 100
        max_delta = 0.02
        found_solution = False
        for i in range(1, max_i+1):
            delta = i * max_delta / max_i
            p_test = generator.get_p(fi, n, delta)
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

    imdata = renderer.get_rgb(counts)
    print(f'Done: {fi} / {n} frame in {time.perf_counter() - t:.3f}s')
    plt.imsave(os.path.join(output_location, f'frame{fi:04d}.gif'), imdata)
    plt.imsave(os.path.join(output_location, f'frame{fi:04d}.png'), imdata)


if __name__ == '__main__':
    if len(sys.argv) <= 6:
        print(f'{sys.argv[0]} <config> <blocks> <frames> <width> <height> <cores>')
        exit(1)
    config_path = sys.argv[1]
    blocks = int(sys.argv[2])
    frames = int(sys.argv[3])
    size = (int(sys.argv[5]), int(sys.argv[4]))
    cores = int(sys.argv[6])

    # Read settings from json file
    with open(config_path, 'r') as f:
        json_data: dict = json.load(f)
        name = json_data["name"]
        if not json_data["type"] == "clifford":
            exit(1)
        generator_settings = clifford.get_generator(json_data["data"]["generation"])
        render_settings = clifford.get_renderer(json_data["data"]["render"])

    print(f'Building {name}.gif')

    # Delete old frames.
    frame_list = glob.glob(os.path.join(output_location, 'frame*.gif'))
    for frame in frame_list:
        os.remove(frame)
    frame_list = glob.glob(os.path.join(output_location, 'frame*.png'))
    for frame in frame_list:
        os.remove(frame)

    # Helper function, so we can just pass a frame index. Makes the multiprocessing clearer.
    def solve_frame_p(x):
        solve_frame(x, blocks, frames, generator_settings, render_settings, size)

    t = time.perf_counter()
    if debug:
        # Just one frame at a time.
        # Also, mp can squash error messages sometimes.
        for i in range(frames):
            solve_frame_p(i)
    else:
        # Iterate over frames on multiple cores.
        with multiprocessing.Pool(cores) as pool:
            pool.imap(solve_frame_p, range(frames))
            pool.close()
            pool.join()
    time_used = time.perf_counter() - t
    print(f'Done: {frames} frames in {int(time_used/60):d} min {time_used % 60:.1f}s')

    # Save stuff
    print(f'Rendering gif')
    convert_out = os.path.join(output_location, f'{name}.gif')
    make_gifsicle(output_location, convert_out)
    gifsicle_cleanup(convert_out, convert_out)
    make_mp4(output_location, os.path.join(output_location, f'{name}.mp4'))
    print(f'Finished')
