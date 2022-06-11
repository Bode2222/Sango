# Implementation Details here: https://robertheaton.com/2018/12/17/wavefunction-collapse-algorithm/
import math
import random
import numpy as np
import time
from Tuple import Tuple2D


class Dir:
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    IN = 4
    OUT = 5

# Describes relationships between two tiles. Tile1, Tile2, left means t1 is allowed to be on the left of t2
class Rule:
    def __init__(self, t1, t2, dir):
        self.t1 = t1
        self.t2 = t2
        self.dir = dir

# WF Collapse interface
class WaveFunctionCollapse:
    def __init__(self, dims, n_tiles, inventory, rules, weights):
        pass
    def place(loc, int):
        pass
    def step():
        pass
    def reset():
        pass
    def _getTupleObject(n_tiles, dims, context_dims):
        pass

# 2D impl of WF Collapse interface
class WFCollapse2D(WaveFunctionCollapse):
    NUM_DIRS = 4

    # inventory maps tile id to number of units available for use. if number of units is -1, that means infinite available
    def __init__(self, dims, n_tiles, rules, strategy= "BASIC", inventory={-1: -1}, weights=[-1], context_dims=[3, 3]):
        self._dims = dims
        self._rules = rules
        self._n_tiles = n_tiles
        self._strategy = strategy
        self._grid = self._getTupleObject(n_tiles, dims, context_dims)
        self._adj = self._gen_adjacency_matrix()

        # Weight handling. if no weights, give even weights for all
        self._weights = weights
        if weights[0] == -1:
            self._weights = [1/n_tiles for i in range(n_tiles)]
        
        # Inventory handling. if not set, set all to infinity. Ti
        self._orig_inv = inventory
        if -1 in inventory.keys():
            self._orig_inv = [-1 for i in range(n_tiles)]
        else:
            self._orig_inv = [inventory[i] if i in inventory.keys() else -1 for i in range(n_tiles)]
        self._inv = self._orig_inv
        
    # Place a cell at a given location
    def place(self, loc, val):
        # Update weights
        self._get_lowest_entropy()
        # Set chosen tile to cell value
        self._grid.get_pos(loc).chosen_tile = val
        # Remove all other tiles from available tile list
        for i in range(len(self._grid.get_pos(loc).tile_active)):
            if i != val:
                self._grid.get_pos(loc).tile_active[i] = False
        self._propagate(loc)

    # Reset entire grid to pre changed form
    def reset(self):
        self._grid = self._getTupleObject(self._n_tiles, self._dims)
        self._inv = self._orig_inv

    def step(self):
        loc = self._get_lowest_entropy()
        if (loc[0] == -1): return 0
        self._collapse(loc)
        self._propagate(loc)
        return 1
        
    # Sets the tuple object to correct subclass of tuple
    def _getTupleObject(self, n_tiles, dims, context_dims):
        return Tuple2D(n_tiles, dims, context_dims)

    def _gen_adjacency_matrix(self):
        res = [[False for ii in range(self._n_tiles)] for i in range(self._n_tiles * self.NUM_DIRS)]
        for r in self._rules:
            res[r.t1 * self.NUM_DIRS + ((r.dir+2) % self.NUM_DIRS)][r.t2] = True
        return np.array(res)

    # When a tile has run out of stock in the inventory remove it from all cells
    # Generate weights and Calculate the entropy for each cell using shannon entropy shown below, and selects the cell with the lowest entropy
    # shannon_entropy_for_square = log(sum(weight)) - (sum(weight * log(weight)) / sum(weight))
    def _get_lowest_entropy(self):
        i = 0
        min_index = -1
        min_entropy = float('inf')

        # Keeps track of all the cells that have the minimum entropy and randomly selects from it
        min_ent_list = []
        for curCell, context in self._grid:
            # If a cell has already chosen a tile, skip it
            if curCell.chosen_tile != -1:
                i += 1
                continue

            sum_weight = 0
            log_sum_weight = 0

            # update self._weights as its needed in the calculation of entropy
            self._set_weights(context)

            no_active_tiles = True
            # Go through all the active tiles that can be chosen and add up their weights
            for ii in range(len(curCell.tile_active)):
                if curCell.tile_active[ii]:
                    if self._inv[ii] == 0:
                        curCell.tile_active[ii] = False
                        continue
                    no_active_tiles = False
                    sum_weight += self._weights[ii]
                    log_sum_weight += self._weights[ii] * math.log(self._weights[ii])

            # If a cell has no tiles that can be chosen, select the 'none' tile and skip it
            if no_active_tiles:
                i += 1
                curCell.chosen_tile = -2
                continue
            # Calculate the entropy from the summed weights
            entropy = math.log(sum_weight) - log_sum_weight / sum_weight
            # compare to minimum
            if entropy < min_entropy:
                min_ent_list = [i]
                min_entropy = entropy
            elif entropy == min_entropy:
                min_ent_list.append(i)
            i += 1

        # If we couldnt calcuate the entropy of a single cell then all the cells have chosen tiles and the program is over
        if min_entropy == float('inf'):
            return [-1]
        
        # chosen cell is randomly selected from list of cells at the minimum entropy
        min_index = random.choice(min_ent_list)

        # Return the location of the cell with min entropy
        # if the order we iterate throught the grid changes this needs to change as well
        if len(self._dims) == 2:
            return [min_index % self._dims[0], int(min_index/self._dims[0])]
        else:
            x = int(min_index/(self._dims[1]*self._dims[0]))
            min_index -= x * self._dims[1] * self._dims[0]
            y = int(min_index/(self._dims[1]))
            z = min_index % self.dims[1]
            return [x, y, z]

    # Randomly select an option from remaining options
    def _collapse(self, loc):
        self._set_weights(self._grid.get_context(loc))
        # Get list of available tiles
        available_tile_ids = [x for x in range(len(self._grid.get_pos(loc).tile_active)) if self._grid.get_pos(loc).tile_active[x]]
        available_weights = [self._weights[x] for x in range(len(self._grid.get_pos(loc).tile_active)) if self._grid.get_pos(loc).tile_active[x]]

        # choose a random tile
        chosen_tile = random.choices(available_tile_ids, weights=available_weights)[0]
        # Set chosen tile to cell value
        self._grid.get_pos(loc).chosen_tile = chosen_tile
        # Remove tile from inventory
        if self._inv[chosen_tile] > 0:
            self._inv[chosen_tile] -= 1

        # Remove all other tiles from available tile list
        for i in range(len(self._grid.get_pos(loc).tile_active)):
            if i != chosen_tile:
                self._grid.get_pos(loc).tile_active[i] = False
        return chosen_tile

    def _propagate(self, loc):
        # Add current cell to the stack
        stack = [loc]
        # While the stack is not empty
        while stack:
            # pop the stack
            pos = stack.pop()
            # If this cell has no tiles to choose from just ignore it
            if np.count_nonzero(self._grid.get_pos(pos).tile_active) == 0:
                continue
            # Go through my available tiles and or their different directional adjaceny tiles.
            north = np.array([False for i in range(self._n_tiles)])
            east = np.array([False for i in range(self._n_tiles)])
            south = np.array([False for i in range(self._n_tiles)])
            west = np.array([False for i in range(self._n_tiles)])

            # If this cell is available to us, logical or its rules into our sum of available tiles for a given direction
            for i in range(self._n_tiles):
                if self._grid.get_pos(pos).tile_active[i]:
                    north = np.logical_or(north, self._adj[i * 4 + Dir.UP])
                    east = np.logical_or(east, self._adj[i * 4 + Dir.RIGHT])
                    south = np.logical_or(south, self._adj[i * 4 + Dir.DOWN])
                    west = np.logical_or(west, self._adj[i * 4 + Dir.LEFT])
            
            # Eliminate neighbor possibilities based on rules
            for x, y in [[0, 1], [1, 0], [0, -1], [-1, 0]]:
                    n_pos = [pos[0] + x, pos[1] + y]
                    if (n_pos[0] < 0 or n_pos[0] >= self._dims[0] or n_pos[1] < 0 or n_pos[1] >= self._dims[1]):
                        continue
                    curCell = self._grid.get_pos(n_pos)
                    # Get adjacency rule based on chosen tile
                    # Go through every tile in the tile_active list and OR the result, then AND that with the corresponding neighbor
                    adj = []
                    if (x == 0 and y == -1 and n_pos[1] >= 0):
                        north &= curCell.tile_active
                        if (curCell.chosen_tile == -1 and not np.allclose(north, curCell.tile_active)):
                            curCell.tile_active = list(north)
                            stack.append(n_pos)
                    elif (x == 1 and y == 0 and n_pos[0] < self._dims[0]):
                        east &= curCell.tile_active
                        if (curCell.chosen_tile == -1 and not np.allclose(east, curCell.tile_active)):
                            curCell.tile_active = list(east)
                            stack.append(n_pos)
                    elif (x == 0 and y == 1 and n_pos[1] < self._dims[1]):
                        south &= curCell.tile_active
                        if (curCell.chosen_tile == -1 and not np.allclose(south, curCell.tile_active)):
                            curCell.tile_active = list(south)
                            stack.append(n_pos)
                    elif (x == -1 and y == 0 and n_pos[0] >= 0):
                        west &= curCell.tile_active
                        if (curCell.chosen_tile == -1 and not np.allclose(west, curCell.tile_active)):
                            curCell.tile_active = list(west)
                            stack.append(n_pos)
                        
    # Generate weights given context
    def _gen_weights(self, context):
        if self._strategy == "BASIC":
            return self._weights
        print("Unknown strategy. Defaulting to BASIC")
        return self._weights

    def _basic_weight_strategy(self, context):
        return self._weights

    # set weights to new values calculated based on context of given cell location
    def _set_weights(self, context):
        self._weights = self._gen_weights(context)

if __name__ == '__main__':
    # Set random seed for consistent results
    random.seed(a=1235)

    # Set generation rules. 0, 1, 2 = Land, sea, coast
    rules = [
        Rule(0, 2, Dir.UP), Rule(0, 2, Dir.DOWN), Rule(0, 2, Dir.LEFT), Rule(0, 2, Dir.RIGHT),
        Rule(2, 0, Dir.UP), Rule(2, 0, Dir.DOWN), Rule(2, 0, Dir.LEFT), Rule(2, 0, Dir.RIGHT),
        Rule(0, 0, Dir.UP), Rule(0, 0, Dir.DOWN), Rule(0, 0, Dir.LEFT), Rule(0, 0, Dir.RIGHT),
        Rule(1, 2, Dir.UP), Rule(1, 2, Dir.DOWN), Rule(1, 2, Dir.LEFT), Rule(1, 2, Dir.RIGHT),
        Rule(2, 1, Dir.UP), Rule(2, 1, Dir.DOWN), Rule(2, 1, Dir.LEFT), Rule(2, 1, Dir.RIGHT),
        Rule(1, 1, Dir.UP), Rule(1, 1, Dir.DOWN), Rule(1, 1, Dir.LEFT), Rule(1, 1, Dir.RIGHT),
        Rule(2, 2, Dir.UP), Rule(2, 2, Dir.DOWN), Rule(2, 2, Dir.LEFT), Rule(2, 2, Dir.RIGHT)
    ]
    # Create Wave-funciton collapse object
    test = WFCollapse2D([10, 10], 3, rules, inventory={0: -1, 1: -1})

    # Keep stepping till generation is done
    start_time = time.time()
    while test.step():
        pass
    print(str(test._grid))
    print("Execution time: %s seconds" % (time.time() - start_time))

    print("WaveFunc Program terminated")