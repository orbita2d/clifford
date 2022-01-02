import matplotlib.pyplot as plt
import clifford
import os.path
import sys
import random

block_size = int(5E6)

output_location = './frames/'
debug = False

blocks = 25
size = (800, 800)

maps = ['Greys',
        'Purples',
        'Blues',
        'Oranges',
        'Reds',
        'OrRd',
        'PuRd',
        'RdPu',
        'BuPu',
        ]

select_options = ['dr', 'dr',
                  'd2',
                  'd2r']


def generate_from_index(index: int):

    random.seed(index)
    print(f'{index:04x}')
    while True:
        generator_data = {
            "type": "frame",
            "data": {
                "p0": [random.uniform( 1.5,  4.0),
                       random.uniform( 1.5,  3.0),
                       random.uniform( 0.8,  2.2),
                       random.uniform(-0.8, -2.2)]
            }
        }
        generator = clifford.get_generator(generator_data)
        p_frame = generator.get_p(0, 0, 0.0)
        if not clifford.test_closed(p_frame):
            break
    print(generator_data)
    map1, map2 = random.sample(maps, 2)
    select, = random.sample(select_options, 1)
    if select == 'dr':
        exponent = random.uniform(0.3, 0.7),
    else:
        exponent = random.uniform(0.9, 1.1),
    renderer_data = {
      "type": "bilinear",
      "data": {
        "map1": map1,
        "map2": map2,
        "lut": 256,
        "select": select,
        "exponent": exponent,
        "invert": False,
        "alpha": 0.3
      }}
    print(renderer_data)
    renderer = clifford.get_renderer(renderer_data)
    counts = clifford.ArrayCounts(size, 0.05)
    x0 = 0.08
    y0 = 0.12
    for block in range(blocks):
        x0, y0 = clifford.get_frame(x0, y0, p_frame, block_size, counts)
    imdata = renderer.get_rgb(counts)
    plt.imsave(os.path.join(output_location, f'index{index:04x}.png'), imdata)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'{sys.argv[0]} <index>')
        exit(1)
    index: int = int(sys.argv[1], 16)
    generate_from_index(index)

