from shutil import copyfile
import os.path
import clifford
import json
import sys

output_location = './frames/'

if __name__ == "__main__":
    config_path = sys.argv[1]

    # Read settings from json file
    with open(config_path, 'r') as f:
        json_data: dict = json.load(f)
        name = json_data["name"]
        if not json_data["type"] == "clifford":
            exit(1)

    copyfile(os.path.join(output_location, f'frame0000.png'), os.path.join(output_location, f'{name}.png'))
    print(f'{name}.png')
    clifford.make_gifsicle(output_location, os.path.join(output_location, f'{name}.gif'))
    print(f'{name}.gif')
    clifford.make_mp4(output_location, os.path.join(output_location, f'{name}.mp4'))
    print(f'{name}.mp4')