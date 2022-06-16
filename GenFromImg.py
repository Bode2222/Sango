import os
import imageio as iio
from Tuple import Tuple2D
import tensorflow as tf
from Tuple import Cell
from WaveFuncCollapse import WFCollapse2D, extractRulesAndRelativeFrequencies2D


# Turns an image into tuple2d
def img_to_tuple(filename: str):
    img = iio.imread(filename)

    curTile = 0
    # mapping from colors to tiles
    color_map = {}
    # resulting tiled image
    grid = [[0 for y in range(len(img[0]))] for x in range(len(img))]
    
    # img is 3d array. hash every new color
    for x in range(len(img)):
        for y in range(len(img[0])):
            col = tuple(img[x][y])
            if col not in color_map.keys():
                color_map[col] = curTile
                curTile += 1
            grid[x][y] = color_map[col]

    # list the colors as they map to the tiles
    col_list = [[] for x in range(curTile)]
    keys = list(color_map.keys())
    vals = list(color_map.values())
    for v in vals:
        col_list[v] = keys[vals.index(v)]

    # convert grid to tuple 2d
    res = Tuple2D(curTile, [len(img), len(img[0])])
    for x in range(len(img)):
        for y in range(len(img[0])):
            res.set_pos([x, y], Cell(curTile, chosen_tile=grid[x][y]))

    return res, col_list



if __name__ == '__main__':
    # Get path to current file
    cur_path = os.path.dirname(__file__)
    # if you wanted to go in the parent dir, append to cur path a seperator and the 'parent_directory' symbol
    #path = os.path.normpath(cur_path + os.sep + os.pardir)
    path = os.path.join(cur_path, "samples\\Flowers.png")
    img_to_tuple(str(path))
    print(path)
    print("Generation from image progam complete")