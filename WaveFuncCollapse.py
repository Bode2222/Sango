# Implementation Details here: https://robertheaton.com/2018/12/17/wavefunction-collapse-algorithm/
import math
import random
from traceback import print_tb
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
        self._context_dims = context_dims
        # init grid
        self._grid = self._getTupleObject(n_tiles, dims, context_dims)
        self._adj = self._gen_adjacency_matrix()

        # Weight entropy storage stuff
        self._min_entropy = float('inf')
        self._min_ent_list = set()
        # Buffer to store weight, entropy pairs for each cell
        self._weight_entropy = [[[], float('inf')] for i in range(len(self._grid))]

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

        # set all the weights in the grid
        self._set_weights([i for i in range(len(self._grid))])
    
    # Place a cell at a given location
    def place(self, loc, val):
        # Update weights
        self._get_lowest_entropy()
        # Set chosen tile to cell value
        self._grid.get_cell(loc).chosen_tile = val
        # Remove all other tiles from available tile list
        for i in range(len(self._grid.get_cell(loc).tile_active)):
            if i != val:
                self._grid.get_cell(loc).tile_active[i] = False
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

    # Calculate all the weight, entropy pairs for all the cells in the grid
    # takes a list of cell indexes
    # shannon_entropy_for_square = log(sum(weight)) - (sum(weight * log(weight)) / sum(weight))
    def _set_weights(self, loc):
        # For every cell in the location list, get its cell and context, calc its weights and entropy, store em
        for i in range(len(loc)):
            pos = self._grid.index_to_loc(loc[i])
            index = loc[i]
            # store the newly calculated weights and entropy
            weights, entropy = self._calc_weight_entropy(pos)
            self._weight_entropy[index][0] = weights
            self._weight_entropy[index][1] = entropy

            # compare to minimum and add to min array accordingly
            if entropy < self._min_entropy or len(self._min_ent_list) == 0:
                self._min_ent_list = [pos]
                self._min_entropy = entropy
            # if it has the same entropy and is not already in the list
            elif entropy == self._min_entropy and pos not in self._min_ent_list:
                self._min_ent_list.append(pos)
            i += 1
        
        if self._min_entropy == float('inf'):
            for i in range(len(self._grid)):
                pos = self._grid.index_to_loc(i)
                weights = self._weight_entropy[i][0]
                entropy = self._weight_entropy[i][1]
                
                # compare to minimum and add to min array accordingly
                if entropy < self._min_entropy:
                    self._min_ent_list = [pos]
                    self._min_entropy = entropy
                # if it has the same entropy and is not already in the list
                elif entropy == self._min_entropy and pos not in self._min_ent_list:
                    self._min_ent_list.append(pos)
        
    # returns weight and entropy for single cell
    def _calc_weight_entropy(self, pos):
        weight = []
        curCell, context = self._grid.get_cell_context(pos)
        # If a cell has already chosen a tile, skip it
        if curCell.chosen_tile != -1:
            return [weight, float('inf')]
        
        weight = self._gen_weights(context, pos)
        sum_weight = 0
        log_sum_weight = 0
        no_active_tiles = True

        # Go through all the active tiles that can be chosen and add up their weights
        for ii in range(len(curCell.tile_active)):
            if curCell.tile_active[ii]:
                # When a tile has run out of stock in the inventory remove it from all cells
                if self._inv[ii] == 0:
                    curCell.tile_active[ii] = False
                    continue
                no_active_tiles = False
                sum_weight += weight[ii]
                log_sum_weight += weight[ii] * math.log(weight[ii])

        # If a cell has no tiles that can be chosen, select the 'none' tile and skip it
        if no_active_tiles:
            curCell.chosen_tile = -2
            return [weight, float('inf')]
            
        # Calculate the entropy from the summed weights
        entropy = math.log(sum_weight) - log_sum_weight / sum_weight
        return [weight, entropy]

    # selects the cell with the lowest entropy
    def _get_lowest_entropy(self):
        #self._set_weights([i for i in range(len(self._grid))])

        # If we couldnt calcuate the entropy of a single cell then all the cells have chosen tiles and the program is over
        if self._min_entropy == float('inf'):
            return [-1]
        
        # chosen cell is randomly selected from list of cells at the minimum entropy and removed from list
        min_loc = self._min_ent_list.pop(random.randrange(len(self._min_ent_list)))

        # Return the location of the cell with min entropy
        return min_loc

    # Randomly select an option from remaining options
    def _collapse(self, loc):
        # Get list of available tiles
        available_tile_ids = [x for x in range(len(self._grid.get_cell(loc).tile_active)) if self._grid.get_cell(loc).tile_active[x]]
        available_weights = [self._weight_entropy[self._grid.loc_to_index(loc)][0][x] for x in range(len(self._grid.get_cell(loc).tile_active)) if self._grid.get_cell(loc).tile_active[x]]

        # choose a random tile
        chosen_tile = random.choices(available_tile_ids, weights=available_weights)[0]
        # Set chosen tile to cell value
        self._grid.get_cell(loc).chosen_tile = chosen_tile
        # Set cell entropy to infinite
        self._weight_entropy[loc[0] * self._grid.wid + loc[1]][1] = float('inf')
        # Remove tile from inventory
        if self._inv[chosen_tile] > 0:
            self._inv[chosen_tile] -= 1

        return chosen_tile

    # after propagating, keep a list of all the cells that changed and update their weights based on their contexts
    def _propagate(self, loc):
        # set of cells who were affected by propagation, i.e set of cells whose contexts were changed and update their weights and entropies
        affected_cells = set()
        # Add current cell to the stack
        stack = [loc]
        # While the stack is not empty
        while stack:
            # pop the stack
            pos = stack.pop()
            # If this cell has no tiles to choose from just ignore it
            if np.count_nonzero(self._grid.get_cell(pos).tile_active) == 0 or self._grid.get_cell(pos).chosen_tile == -2:
                continue

            # add the context of this tile into the affected cells pile
            lx = int(self._context_dims[0]/2)
            ly = int(self._context_dims[1]/2)
            affected_cells.update(list(map(self._grid.loc_to_index, self._grid.get_context_positions(pos))))

            # Get the list of tiles allowed beside current tile: Go through my available tiles and 'or' their different directional adjaceny tiles.
            north = south = east = west = []
            if self._grid.get_cell(pos).chosen_tile != -1:
                north = np.array(self._adj[self._grid.get_cell(pos).chosen_tile * 4 + Dir.UP])
                east = np.array(self._adj[self._grid.get_cell(pos).chosen_tile * 4 + Dir.RIGHT])
                south = np.array(self._adj[self._grid.get_cell(pos).chosen_tile * 4 + Dir.DOWN])
                west = np.array(self._adj[self._grid.get_cell(pos).chosen_tile * 4 + Dir.LEFT])
            else:
                north = np.array([False for i in range(self._n_tiles)])
                east = np.array([False for i in range(self._n_tiles)])
                south = np.array([False for i in range(self._n_tiles)])
                west = np.array([False for i in range(self._n_tiles)])
                # If this cell is available to us, logical or its rules into our sum of available tiles for a given direction
                for i in range(self._n_tiles):
                    if self._grid.get_cell(pos).tile_active[i]:
                        north = np.logical_or(north, self._adj[i * 4 + Dir.UP])
                        east = np.logical_or(east, self._adj[i * 4 + Dir.RIGHT])
                        south = np.logical_or(south, self._adj[i * 4 + Dir.DOWN])
                        west = np.logical_or(west, self._adj[i * 4 + Dir.LEFT])
            
            # Eliminate neighbor possibilities based on rules
            for x, y in [[0, 1], [1, 0], [0, -1], [-1, 0]]:
                    n_pos = [pos[0] + x, pos[1] + y]
                    if (n_pos[0] < 0 or n_pos[0] >= self._dims[0] or n_pos[1] < 0 or n_pos[1] >= self._dims[1]):
                        continue
                    curCell = self._grid.get_cell(n_pos)
                    # Get adjacency rule based on chosen tile. If the permutation of tile active changes add it to the stack
                    # Go through every tile in the tile_active list and OR the result, then AND that with the corresponding neighbor
                    adj = []
                    if (y == -1 and n_pos[1] >= 0):
                        north &= curCell.tile_active
                        if (curCell.chosen_tile == -1 and not np.allclose(north, curCell.tile_active)):
                            curCell.tile_active = list(north)
                            stack.append(n_pos)
                    elif (x == 1 and n_pos[0] < self._dims[0]):
                        east &= curCell.tile_active
                        if (curCell.chosen_tile == -1 and not np.allclose(east, curCell.tile_active)):
                            curCell.tile_active = list(east)
                            stack.append(n_pos)
                    elif (y == 1 and n_pos[1] < self._dims[1]):
                        south &= curCell.tile_active
                        if (curCell.chosen_tile == -1 and not np.allclose(south, curCell.tile_active)):
                            curCell.tile_active = list(south)
                            stack.append(n_pos)
                    elif (x == -1 and n_pos[0] >= 0):
                        west &= curCell.tile_active
                        if (curCell.chosen_tile == -1 and not np.allclose(west, curCell.tile_active)):
                            curCell.tile_active = list(west)
                            stack.append(n_pos)
        
        # change cell indexes back to locations, make them into a list then recalculate every affected cell
        self._set_weights(list(affected_cells))
                        
    # Generate weights given context
    def _gen_weights(self, context, pos):
        if self._strategy == "BASIC":
            return self._basic_weight_strategy(context, pos)
        print("Unknown strategy. Defaulting to BASIC")
        return self._weights

    def _basic_weight_strategy(self, context, pos):
        return self._weights

# Given a grid, iterate over it and add every rule found in it
def extractRules2D(grid: Tuple2D):
    # Set to place results
    result = set()
    for x in range(grid.len):
        for y in range(grid.wid):
            # For every adjacent tile
            #if (x + 1)
            pass

            pass
    pass

# TODO: Implement context generation strategy
# TODO: Make function to turn tuple2d into list of rules
# TODO: Add a price system and put in the prices of each tiles. Use this to determine which tiles can be placed during tile selection. 
# this means each step if the amount of currency possessed has changed (increased past the next most expensive item or decreased lower than the current most expensive) we need to update the entire boards weights
# TODO: Use my gpu to do entropy calculation somehow?
# TODO: Make our ai powered result work with the 'Terminal' game
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
    test = WFCollapse2D(dims=[28, 28], n_tiles=3, rules=rules, inventory={0: -1, 1: -1})

    # Keep stepping till generation is done
    start_time = time.time()
    while test.step():  pass
    stop_time = time.time()
    print(str(test._grid))
    print("Execution time: %s seconds" % (stop_time - start_time))

    print("WaveFunc Program terminated")