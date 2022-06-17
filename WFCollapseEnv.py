from cgi import test
from Tuple1D import Tuple, Tuple1D
from WaveFuncCollapse import WaveFunctionCollapse


# State object holds a tuple and a current location
class State:
    def __init__(self, tuple: Tuple, loc) -> None:
        self._tuple = tuple
        self._loc = loc
    
    def get(self):
        return [self._tuple, self._loc]

# an action object holds a location and a chosen tile
class Action:
    def __init__(self, loc, chosen_tile) -> None:
        self._loc = loc
        self._chosen = chosen_tile
    
    def get(self):
        return [self._chosen, self._loc]

# wf collapse algo, but reimagined as an environment for reinforcemnet learning
# None of these are specific to 1 dimension or a specific game. this should be an abstract class other classes can inherit from
class WFCollapseEnv(WaveFunctionCollapse):
    def __init__(self, dims, n_tiles, rules=[], tile_selection="LINEAR", weight_genner=None, inventory={ -1: -1 }, context_space="LOCAL", context_dims=[3, 3], prices={ -1: 0 }, money=10):
        tile_selection="LINEAR"
        super().__init__(dims, n_tiles, rules, tile_selection, weight_genner, inventory, context_space, context_dims, prices, money)

    # reset now returns the current state of the board
    def reset(self):
        super().reset()
        loc = self._get_lowest_entropy()
        return State(self._grid.get_cell_context(loc), loc)

    # step now takes in an 'action' which is the tile id chosen to be in that space
    def step(self, action: Action):
        loc, chosen_tile = action.get()
        self.place(loc, chosen_tile)
        self._propagate()
        # Return next state and reward
        n_loc = self._get_lowest_entropy()
        return [State(self._grid.get_cell_context(n_loc), n_loc), self.reward()]

    # define reward function here
    def reward(self):
        return 0

if __name__ == '__main__':
    print("WF Env program finished")