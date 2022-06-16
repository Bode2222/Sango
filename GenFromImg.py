# Holds functions required to take an image and extract from it necessary features to use wave func collapse

import os
import imageio as iio
from Tuple2D import Tuple2D
from Tuple import Cell
from WFUtil import Rule, Dir


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

# Given a grid, iterate over it and add every rule found in it as well as the frequency of each tile regardless of the tiles around it.
# The frequency output is used in basic weight gen
def extractRulesAndTotalFrequencies2D(grid: Tuple2D):
	# Set to place results
	result = set()
	tile_frequency = []
	seen_tiles = {}
	for x in range(grid.len):
		for y in range(grid.wid):
			# check this tile and its neighbors
			# if we havent seen them before, register it and start counting their occurences. if we have just increase its occurences
			c_t = grid.get_cell([x, y]).chosen_tile
			c_t_n = grid.get_cell([x, y-1]).chosen_tile
			c_t_e = grid.get_cell([x + 1, y]).chosen_tile
			c_t_w = grid.get_cell([x, y + 1]).chosen_tile
			c_t_s = grid.get_cell([x - 1, y]).chosen_tile
			for chosen in [c_t, c_t_n, c_t_e, c_t_s, c_t_w]:
				if chosen not in seen_tiles.keys():
					seen_tiles[chosen] = len(tile_frequency)
					tile_frequency.append(1)
				else:
					tile_frequency[seen_tiles[chosen]] += 1

			# Check around it to get its rules
			if y > 0:
				result.add(Rule(seen_tiles[c_t], seen_tiles[c_t_n], Dir.UP))
				result.add(Rule(seen_tiles[c_t_n], seen_tiles[c_t], Dir.DOWN))
			if x < grid.len:
				result.add(Rule(seen_tiles[c_t], seen_tiles[c_t_e], Dir.RIGHT))
				result.add(Rule(seen_tiles[c_t_e], seen_tiles[c_t], Dir.LEFT))
			if y < grid.wid:
				result.add(Rule(seen_tiles[c_t], seen_tiles[c_t_s], Dir.DOWN))
				result.add(Rule(seen_tiles[c_t_s], seen_tiles[c_t], Dir.UP))
			if x > 0:
				result.add(Rule(seen_tiles[c_t], seen_tiles[c_t_w], Dir.RIGHT))
				result.add(Rule(seen_tiles[c_t_w], seen_tiles[c_t], Dir.LEFT))
	return [list(result), tile_frequency, seen_tiles]

# for every cell in every direction find the frequency each tile appears
# Used in Simple Weight gen
def extractRulesAndRelativeFrequencies2D(grid: Tuple2D):
	# Set to place results
	result = set()
	# map a tile and a direction to the frequency that ever tile appears in that direction
	# every time a new tile appears it grows longer by 4 and every list in it grows longer by one
	tile_frequency = []
	seen_tiles = {}
	for x in range(grid.len):
		for y in range(grid.wid):
			# check this tile and its neighbors
			# if we havent seen them before, register it and start counting their occurences. if we have just increase its occurences
			c_t = grid.get_cell([x, y]).chosen_tile
			c_t_n = grid.get_cell([x, y-1]).chosen_tile
			c_t_e = grid.get_cell([x + 1, y]).chosen_tile
			c_t_w = grid.get_cell([x, y + 1]).chosen_tile
			c_t_s = grid.get_cell([x - 1, y]).chosen_tile
			c_t_l = [c_t, c_t_n, c_t_e, c_t_s, c_t_w]
			for i in range(len(c_t_l)):
				chosen = c_t_l[i]
				# if we havent seen this tile before
				if chosen not in seen_tiles.keys():
					# grow longer by 4 and set all the new lists to 0
					# Number of tiles seen is the length of the freq array divided by 4
					cur_len = int(len(tile_frequency)/4)
					seen_tiles[chosen] = cur_len
					# add frequency arrays for each directions
					for freq_arr in tile_frequency:
						freq_arr.append(0)
					tile_frequency.append([0 for x in range(cur_len + 1)])
					tile_frequency.append([0 for x in range(cur_len + 1)])
					tile_frequency.append([0 for x in range(cur_len + 1)])
					tile_frequency.append([0 for x in range(cur_len + 1)])
				# Add its neighbors to the corresponding array
				if i == 1:
					tile_frequency[seen_tiles[c_t] * 4 + Dir.UP][seen_tiles[c_t_n]] += 1
					tile_frequency[seen_tiles[c_t_n] * 4 + Dir.DOWN][seen_tiles[c_t]] += 1
				elif i == 2:
					tile_frequency[seen_tiles[c_t] * 4 + Dir.RIGHT][seen_tiles[c_t_e]] += 1
					tile_frequency[seen_tiles[c_t_e] * 4 + Dir.LEFT][seen_tiles[c_t]] += 1

			# Check around it to get its rules
			if y > 0:
				result.add(Rule(seen_tiles[c_t], seen_tiles[c_t_n], Dir.UP))
				result.add(Rule(seen_tiles[c_t_n], seen_tiles[c_t], Dir.DOWN))
			if x < grid.len:
				result.add(Rule(seen_tiles[c_t], seen_tiles[c_t_e], Dir.RIGHT))
				result.add(Rule(seen_tiles[c_t_e], seen_tiles[c_t], Dir.LEFT))
			if y < grid.wid:
				result.add(Rule(seen_tiles[c_t], seen_tiles[c_t_s], Dir.DOWN))
				result.add(Rule(seen_tiles[c_t_s], seen_tiles[c_t], Dir.UP))
			if x > 0:
				result.add(Rule(seen_tiles[c_t], seen_tiles[c_t_w], Dir.RIGHT))
				result.add(Rule(seen_tiles[c_t_w], seen_tiles[c_t], Dir.LEFT))

	tile_frequency = convert_freq_to_orig_form(tile_frequency, seen_tiles)

	# change rules based on mapping
	results = convert_rules_to_orig_form(list(result), seen_tiles)
	# remove any rules for null cells
	results = [rule for rule in results if rule.t1 != -2 and rule.t2 != -2]
	return [results, tile_frequency, seen_tiles]

# Given rules and a mapping use the mapping to convert the rules
def convert_rules_to_orig_form(rules, mapping):
	keys = list(mapping.keys())
	vals = list(mapping.values())
	res = []
	for rule in rules:
		origt1 = keys[vals.index(rule.t1)]
		origt2 = keys[vals.index(rule.t2)]
		res.append(Rule(origt1, origt2, rule.dir))
	return res

# giving frequncies and a mapping use the mapping to rerganize the frequencies
def convert_freq_to_orig_form(freq, mapping):
	keys = list(mapping.keys())
	vals = list(mapping.values())
	res = [[0 for x in range(len(freq[0])-1)] for x in range(len(freq))]

	# first go into individual rows and fix those
	for ii in range(len(freq)):
		tile_dir = freq[ii]
		for i in range(len(tile_dir)):
			orig_tid = keys[vals.index(i)]
			# if this a null cell move all its frequencies to the back of the array
			if orig_tid == -2: continue
			res[ii][orig_tid] = freq[ii][i]

	restwo = [[0 for x in range(len(freq[0]))] for x in range(len(freq) - 4)]
	# then move 4 rows at a time around to their right spots
	for i in range(int(len(freq)/4)):
		orig_tid = keys[vals.index(i)]
		# if this a null cell move all its frequencies to the back of the array
		if orig_tid == -2: 
			orig_tid = -1
			continue

		restwo[orig_tid * 4] = res[i * 4]
		restwo[orig_tid * 4 + 1] = res[i * 4 + 1]
		restwo[orig_tid * 4 + 2] = res[i * 4 + 2]
		restwo[orig_tid * 4 + 3] = res[i * 4 + 3]

	return restwo


if __name__ == '__main__':
    # Get path to current file
    cur_path = os.path.dirname(__file__)
    # if you wanted to go in the parent dir, append to cur path a seperator and the 'parent_directory' symbol
    #path = os.path.normpath(cur_path + os.sep + os.pardir)
    path = os.path.join(cur_path, "samples\\Flowers.png")
    img_to_tuple(str(path))
    print(path)
    print("Generation from image progam complete")