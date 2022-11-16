### Simple_Image_Stitcher

To run:
```
python Q1.py --input_folder {path to folder of images}
```

example
```
python Q1.py --input_folder ./fumo
```

You can view the final images in this repo too.

This script uses opencv's blob detector. Params have to be set accordingly else it will not work.
The other method used was using Hough circle, but it is not as accurate in detecting the circles/dots. 
