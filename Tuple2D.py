from Tuple import Tuple, Cell


class Tuple2D(Tuple):
    # given number of tiles a cell can become, dimensions of grid, and dimensions of area around a cell to give as context
    # strat or strategy determines which context should be returned when asked
    def __init__(self, n_tiles, dims, context_dims=[3, 3], context_space="LOCAL"):
        self.len = dims[0]
        self.wid = dims[1]
        self._strat = context_space
        self.grid = [[Cell(n_tiles) for i in range(self.wid)] for ii in range(self.len)]
        self.con_dims = context_dims

        # Generate context for each cell and store it in a list
        self._con_list = self._generate_contexts()
        self._global_list = [[x, y] for y in range(self.wid) for x in range(self.len)]

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
        return list(map(self._generate_context_pos, locs))

    # Get list of context cell positions given a location
    def _generate_context_pos(self, loc):
        result = []

        for i in range(self.con_dims[1]):
            for ii in range(self.con_dims[0]):
                # Get offsets from 'loc' by subtracting half the context dimensions from that position
                y = i - int(self.con_dims[1]/2)
                x = ii - int(self.con_dims[0]/2)
                pos = [loc[0] + x, loc[1] + y]

                # if its in range replace it with the appropriate cell, else ignore it
                if (pos[0] >= 0 and pos[1] >= 0 and pos[0] < self.len and pos[1] < self.wid):
                    result.append([pos[0], pos[1]])
        return result

    # Get list of context cell locations given a location
    def get_cell_context_positions(self, loc):
        if self._strat == "LOCAL":
            return self._con_list[self.loc_to_index(loc)]
        elif self._strat == "GLOBAL":
            return self._global_list

    # Set a location on the grid to a val
    def set_pos(self, loc, val):
        self.grid[loc[0]][loc[1]] = val

    # Return either an empty cell or the requested cell depending on whether the requested location was within bounds
    def get_cell(self, loc):
        if (loc[0] >= 0 and loc[1] >= 0 and loc[0] < self.len and loc[1] < self.wid):
            return self.grid[loc[0]][loc[1]]
        else:
            return Cell.Null()

    # Returns both cell and context of given location.
    # The get cell context that matters
    def get_cell_context(self, loc):
        if (loc[0] >= 0 and loc[1] >= 0 and loc[0] < self.len and loc[1] < self.wid):
            return [self.grid[loc[0]][loc[1]], list(map(self.get_cell, self.get_cell_context_positions(loc)))]
        else:
            return [Cell.Null(), []]

    def loc_to_index(self, loc):
        return loc[0] * self.wid + loc[1]

    def index_to_loc(self, index):
        return [int(index/self.wid), index%self.wid]

if __name__ == '__main__':
    grid = Tuple2D(0, [7, 5])
    print(grid.loc_to_index([2, 2]))
    print(grid.index_to_loc(grid.loc_to_index([2, 2])))
    print(grid._generate_context_pos([0, 0]))
    print("Tuple program Terminated")