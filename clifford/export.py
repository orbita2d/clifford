import glob
import imageio as iio
import pygifsicle
import os.path


def make_gifsicle(images_path: str, output: str):
    """ Make a gif."""
    frame_list = sorted(glob.glob(os.path.join(images_path, 'frame*.png')))
    with iio.get_writer(output, mode='I', fps=50) as writer:
        for frame in frame_list:
            writer.append_data(iio.imread(frame))
    pygifsicle.optimize(output)


def make_mp4(images_path: str, output: str):
    """ Make an mp4 from <images_path>/frame%0d.png frames. Save to <output>"""
    frame_list = sorted(glob.glob(os.path.join(images_path, 'frame*.png')))
    with iio.get_writer(output, fps=50, quality=6.5) as writer:
        for frame in frame_list:
            writer.append_data(iio.imread(frame))
