from unittest import result
import numpy as np
from Tuple1D import Tuple1D
from WFUtil import Dir
from WaveFuncCollapse import WaveFunctionCollapse
# for testing
import random
from WFUtil import Rule

# 2D impl of WF Collapse interface
class WFCollapse1D(WaveFunctionCollapse):
	NUM_DIRS = 2
	RIGHT = 0
	LEFT = 1
	# Sets the tuple object to correct subclass of tuple
	def _getTupleObject(self, n_tiles, dims, context_dims, context_space):
		return Tuple1D(n_tiles, dims, context_dims, context_space=context_space)

	def _gen_adjacency_matrix(self):
		res = [[False for ii in range(self._n_tiles)] for i in range(self._n_tiles * self.NUM_DIRS)]
		# in here the direction right is 0 and left is 1
		for r in self._rules:
			if (r.dir == Dir.LEFT):
				res[r.t1 * self.NUM_DIRS + self.RIGHT][r.t2] = True
			elif (r.dir == Dir.RIGHT):
				res[r.t1 * self.NUM_DIRS + self.LEFT][r.t2] = True
		return np.array(res)
	
	def _update_neighbors(self, pos):
		results = []
		# Get the list of tiles allowed beside current tile: Go through my available tiles and 'or' their different directional adjaceny tiles.
		east = west = []
		if self._grid.get_cell(pos).chosen_tile != -1:
			east = np.array(self._adj[self._grid.get_cell(pos).chosen_tile * self.NUM_DIRS + self.RIGHT])
			west = np.array(self._adj[self._grid.get_cell(pos).chosen_tile * self.NUM_DIRS + self.LEFT])
		else:
			east = np.array([False for i in range(self._n_tiles)])
			west = np.array([False for i in range(self._n_tiles)])
			# If this cell is available to us, logical or its rules into our sum of available tiles for a given direction
			for i in range(self._n_tiles):
				if self._grid.get_cell(pos).tile_active[i]:
					east = np.logical_or(east, self._adj[i * self.NUM_DIRS + self.RIGHT])
					west = np.logical_or(west, self._adj[i * self.NUM_DIRS + self.LEFT])
		
		# Eliminate neighbor possibilities based on rules
		for x in [1, -1]:
				n_pos = [pos[0] + x]
				if (n_pos[0] < 0 or n_pos[0] >= self._dims[0]):
					continue
				curCell = self._grid.get_cell(n_pos)
				# Get adjacency rule based on chosen tile. If the permutation of tile active changes add it to the stack
				# Go through every tile in the tile_active list and OR the result, then AND that with the corresponding neighbor
				if (x == 1 and n_pos[0] < self._dims[0]):
					east &= curCell.tile_active
					if (curCell.chosen_tile == -1 and not np.allclose(east, curCell.tile_active)):
						curCell.tile_active = list(east)
						results.append(n_pos)
				elif (x == -1 and n_pos[0] >= 0):
					west &= curCell.tile_active
					if (curCell.chosen_tile == -1 and not np.allclose(west, curCell.tile_active)):
						curCell.tile_active = list(west)
						results.append(n_pos)
		return results
		

if __name__ == '__main__':
	#random.seed(123)
	rules = [
		Rule(0, 2, Dir.UP), Rule(0, 2, Dir.DOWN), Rule(0, 2, Dir.LEFT), Rule(0, 2, Dir.RIGHT),
		Rule(2, 0, Dir.UP), Rule(2, 0, Dir.DOWN), Rule(2, 0, Dir.LEFT), Rule(2, 0, Dir.RIGHT),
		Rule(0, 0, Dir.UP), Rule(0, 0, Dir.DOWN), Rule(0, 0, Dir.LEFT), Rule(0, 0, Dir.RIGHT),
		Rule(1, 2, Dir.UP), Rule(1, 2, Dir.DOWN), Rule(1, 2, Dir.LEFT), Rule(1, 2, Dir.RIGHT),
		Rule(2, 1, Dir.UP), Rule(2, 1, Dir.DOWN), Rule(2, 1, Dir.LEFT), Rule(2, 1, Dir.RIGHT),
		Rule(1, 1, Dir.UP), Rule(1, 1, Dir.DOWN), Rule(1, 1, Dir.LEFT), Rule(1, 1, Dir.RIGHT),
		Rule(2, 2, Dir.UP), Rule(2, 2, Dir.DOWN), Rule(2, 2, Dir.LEFT), Rule(2, 2, Dir.RIGHT)
	]
	test1D = WFCollapse1D(dims=[25], n_tiles=3, rules=rules)
	while test1D.step(): pass
	print(test1D._grid)

