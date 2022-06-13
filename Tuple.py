# Grids are made of cells and tiles go in those cells
# -2 means cell with no prospects, -1 means cell with prospects, 0 and above means a tile has been chosen
from difflib import context_diff


class Cell:
    def __init__(self, n_tiles):
        self.tile_active = [True for i in range(n_tiles)]
        self.chosen_tile = -1
    
    def __str__(self):
        return str(self.chosen_tile)

    # Return a 'null' cell
    def Null():
        result = Cell(0)
        result.chosen_tile = -2
        return result

# A tuple takes in the number of tiles that can be placed in any given position to help with cell init
class Tuple:
    def __init__(self, n_tiles, dims, context_dims):
        pass
    def set_pos(self, loc, val):
        pass
    def get_pos(self, loc):
        pass
    def get_context(self, loc):
        pass

class Tuple2D(Tuple):
    # given number of tiles a cell can become, dimensions of grid, and dimensions of area around a cell to give as context
    def __init__(self, n_tiles, dims, context_dims=[3, 3]):
        self.len = dims[0]
        self.wid = dims[1]
        self.grid = [[Cell(n_tiles) for i in range(self.wid)] for ii in range(self.len)]

        self.con_dims = context_dims

        # Generate context for each cell and store it in a list
        self._con_list = self._generate_contexts()

    # return each cell and the cells that surrond it  within the dimensions 'self._con_dims'
    def __iter__(self):
        for ii in range(self.wid):
            for i in range(self.len):
                yield [self.grid[i][ii], self._con_list[i * self.wid + ii]]

    def __len__(self):
        return self.len * self.wid

    def __str__(self):
        res = ""
        for i in range(self.wid):
            for ii in range(self.len):
                res += str(self.grid[ii][i]) + " "
            res += "\n"
        return res

    # returns contexts for all locations in the grid as a list
    def _generate_contexts(self):
        locs = [[x, y] for x in range(self.len) for y in range(self.wid)]
        return list(map(self._calc_context, locs))

    # Get context for a single location
    def _calc_context(self, loc):
        result = [[Cell.Null() for i in range(self.con_dims[1])] for ii in range(self.con_dims[0])]

        for i in range(self.con_dims[1]):
            for ii in range(self.con_dims[0]):
                # Get offsets from 'loc' by subtracting half the context dimensions from that position
                y = i - int(self.con_dims[1]/2)
                x = ii - int(self.con_dims[0]/2)

                # if its in range replace it with the appropriate cell, else ignore it
                result[ii][i] = self.get_pos([loc[0] + x, loc[1] + y])
        return result

    def set_pos(self, loc, val):
        self.grid[loc[0]][loc[1]] = val

    # Return either an empty cell or the requested cell depending on whether the requested location was within bounds
    def get_pos(self, loc):
        if (loc[0] >= 0 and loc[1] >= 0 and loc[0] < self.len and loc[1] < self.wid):
            return self.grid[loc[0]][loc[1]]
        else:
            return Cell.Null()

    def get_pos_context(self, loc):
        if (loc[0] >= 0 and loc[1] >= 0 and loc[0] < self.len and loc[1] < self.wid):
            return [self.grid[loc[0]][loc[1]], self._con_list[loc[0] * self.wid + loc[1]]]
        else:
            return [Cell.Null(), []]

    def get_context(self, loc):
        if (loc[0] >= 0 and loc[1] >= 0 and loc[0] < self.len and loc[1] < self.wid):
            return self._con_list[loc[0] * self.wid + loc[1]]
        else:
            return []

    def loc_to_index(self, loc):
        return loc[0] * self.wid + loc[1]

    def index_to_loc(self, index):
        return [int(index/self.wid), index%self.wid]

if __name__ == '__main__':
    grid = Tuple2D(0, [7, 5])
    print(grid.loc_to_index([2, 2]))
    print(grid.index_to_loc(grid.loc_to_index([2, 2])))
    print("Tuple program Terminated")