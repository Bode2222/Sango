from Tuple import Tuple, Cell

class Tuple1D(Tuple):
    def __init__(self, n_tiles, dims, context_dims=[3], context_space="LOCAL"):
        self.len = dims[0]
        self._strat = context_space
        self.grid = [Cell(n_tiles) for ii in range(self.len)]
        self.con_dims = context_dims

        self._con_list = self._generate_contexts()
        self._global_list = [[x] for x in range(self.len)]

    def set_pos(self, loc, cell: Cell):
        self.grid[loc[0]] = cell

    def get_cell(self, loc) -> Cell:
        return self.grid[loc[0]]

    def get_cell_context(self, loc):
        return [self.grid[loc[0]], [self.grid[x[0]] for x in self.get_cell_context_positions(loc)]]

    # Get list of context cell locations given a location
    def get_cell_context_positions(self, loc):
        if self._strat == "LOCAL":
            return self._con_list[self.loc_to_index(loc)]
        elif self._strat == "GLOBAL":
            return self._global_list

    def loc_to_index(self, loc):
        return loc[0]

    def index_to_loc(self, index):
        return [index]

    # return each cell and the cells that surrond it  within the dimensions 'self._con_dims'
    def __iter__(self):
        half = int(self.con_dims/2)
        for i in range(self.len):
            yield [self.grid[i], self.grid[i - half : i + half]]

    def __len__(self):
        return self.len

    def __str__(self):
        res = ""
        for ii in range(self.len):
            res += str(self.grid[ii]) + " "
        return res

    # returns contexts for all locations in the grid as a list
    def _generate_contexts(self):
        locs = [[x] for x in range(self.len)]
        return list(map(self._generate_context_pos, locs))

    # Get list of context cell positions given a location
    def _generate_context_pos(self, loc):
        result = []

        for ii in range(self.con_dims[0]):
            # Get offsets from 'loc' by subtracting half the context dimensions from that position
            x = ii - int(self.con_dims[0]/2)
            pos = [loc[0] + x]

            # if its in range replace it with the appropriate cell, else ignore it
            if (pos[0] >= 0 and pos[0] < self.len):
                result.append([pos[0]])
        return result

# Modified version of tuple1d whose context is not the cells around it, but the cells behind it
class Tuple1DBackwards(Tuple1D):
    # Get list of context cell positions given a location
    def _generate_context_pos(self, loc):
        result = []

        for ii in range(self.con_dims[0]):
            # Get offsets from 'loc' by subtracting the context dimension from that position
            pos = [loc[0] - ii]

            # if its in range replace it with the appropriate cell, else ignore it
            if (pos[0] >= 0 and pos[0] < self.len):
                result.append([pos[0]])
        return result

if __name__=='__main__':
    test1D = Tuple1D(1, [5])
    cell, context = test1D.get_cell_context([2])
    print(context)

    print("Tuple1D program execution finished")