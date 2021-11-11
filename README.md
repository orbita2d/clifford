# clifford
Clifford attractors are pretty.

### Usage
```
python clifford.py <config.json> <iterations> <frames> <width> <height> <cores>
```
To generate the attractor descibed in erebus.json, with 5 iterations, 200 frames at 360 x 360 resolution on 8 cores:
```
python clifford.py examples/erebus.json 5 200 360 360 8
```

If you need to remove failed frames from the output, you can delete the frame%04d.png images you need and then run 
```
python save.py examples/erebus.json
```
to regenerate the gif and mp4 file.

