# Grids are made of cells and tiles go in those cells
# -2 means cell with no prospects, -1 means cell with prospects, 0 and above means a tile has been chosen


class Cell:
    def __init__(self, n_tiles, chosen_tile=-1):
        self.tile_active = [True for i in range(n_tiles)]
        self.chosen_tile = chosen_tile
    
    def __str__(self):
        return "%02d" % self.chosen_tile

    # Return a 'null' cell
    def Null():
        result = Cell(0)
        result.chosen_tile = -2
        return result

# A tuple takes in the number of tiles that can be placed in any given position to help with cell init
class Tuple:
    def __init__(self, n_tiles, dims, context_dims):
        pass
    def set_pos(self, loc, cell: Cell):
        pass
    def get_cell(self, loc) -> Cell:
        pass
    # Returns both cell and context of given location
    def get_cell_context(self, loc):
        pass
     # Get list of context cell locations given a location
    def get_cell_context_positions(self, loc):
        pass
    def loc_to_index(self, loc):
        pass
    def index_to_loc(self, index):
        pass