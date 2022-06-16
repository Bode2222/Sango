import random
import numpy as np
from Tuple2D import Tuple2D
from WFUtil import Rule, Dir
from WaveFuncCollapse import WaveFunctionCollapse
from BasicWeightGen import BasicWeightGen
from SimpleWeightGen import SimpleWeightGen
from GenFromImg import extractRulesAndRelativeFrequencies2D

# 2D impl of WF Collapse interface
class WFCollapse2D(WaveFunctionCollapse):
	NUM_DIRS = 4
	UP = 0
	RIGHT = 1
	DOWN = 2
	LEFT = 3
	# Sets the tuple object to correct subclass of tuple
	def _getTupleObject(self, n_tiles, dims, context_dims, context_space):
		return Tuple2D(n_tiles, dims, context_dims, context_space=context_space)

	def _gen_adjacency_matrix(self):
		res = [[False for ii in range(self._n_tiles)] for i in range(self._n_tiles * self.NUM_DIRS)]
		# in here up right down left are 0, 1, 2, 3 respectively
		for r in self._rules:
			if (r.dir == Dir.LEFT):
				res[r.t1 * self.NUM_DIRS + self.RIGHT][r.t2] = True
			elif (r.dir == Dir.RIGHT):
				res[r.t1 * self.NUM_DIRS + self.LEFT][r.t2] = True
			elif (r.dir == Dir.UP):
				res[r.t1 * self.NUM_DIRS + self.DOWN][r.t2] = True
			elif (r.dir == Dir.DOWN):
				res[r.t1 * self.NUM_DIRS + self.UP][r.t2] = True
		return np.array(res)
	
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
			if not self._price_change:
				affected_cells.update(list(map(self._grid.loc_to_index, self._grid.get_cell_context_positions(pos))))

			if len(self._rules) > 0:
				# Get the list of tiles allowed beside current tile: Go through my available tiles and 'or' their different directional adjaceny tiles.
				north = south = east = west = []
				if self._grid.get_cell(pos).chosen_tile != -1:
					north = np.array(self._adj[self._grid.get_cell(pos).chosen_tile * self.NUM_DIRS + self.UP])
					east = np.array(self._adj[self._grid.get_cell(pos).chosen_tile * self.NUM_DIRS + self.RIGHT])
					south = np.array(self._adj[self._grid.get_cell(pos).chosen_tile * self.NUM_DIRS + self.DOWN])
					west = np.array(self._adj[self._grid.get_cell(pos).chosen_tile * self.NUM_DIRS + self.LEFT])
				else:
					north = np.array([False for i in range(self._n_tiles)])
					east = np.array([False for i in range(self._n_tiles)])
					south = np.array([False for i in range(self._n_tiles)])
					west = np.array([False for i in range(self._n_tiles)])
					# If this cell is available to us, logical or its rules into our sum of available tiles for a given direction
					for i in range(self._n_tiles):
						if self._grid.get_cell(pos).tile_active[i]:
							north = np.logical_or(north, self._adj[i * self.NUM_DIRS + self.UP])
							east = np.logical_or(east, self._adj[i * self.NUM_DIRS + self.RIGHT])
							south = np.logical_or(south, self._adj[i * self.NUM_DIRS + self.DOWN])
							west = np.logical_or(west, self._adj[i * self.NUM_DIRS + self.LEFT])
				
				# Eliminate neighbor possibilities based on rules
				for x, y in [[0, 1], [1, 0], [0, -1], [-1, 0]]:
						n_pos = [pos[0] + x, pos[1] + y]
						if (n_pos[0] < 0 or n_pos[0] >= self._dims[0] or n_pos[1] < 0 or n_pos[1] >= self._dims[1]):
							continue
						curCell = self._grid.get_cell(n_pos)
						# Get adjacency rule based on chosen tile. If the permutation of tile active changes add it to the stack
						# Go through every tile in the tile_active list and OR the result, then AND that with the corresponding neighbor
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
			
		# if the price crossed a tile price update entire board
		if self._price_change:
			affected_cells = [x for x in range(len(self._grid))]
			self._price_change = False
		# change cell indexes back to locations, make them into a list then recalculate every affected cell
		self._set_weights(list(affected_cells))


# TODO: Use my gpu to do entropy calculation somehow?
# TODO: Make our ai powered result work with the 'Terminal' game
if __name__ == '__main__': 
	# Set random seed for consistent results
	random.seed(a=1235)

	# Grid dimensions
	dims = [35, 35]
	# context dimensions
	con_dims = [3, 3]
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
	test = WFCollapse2D(dims=dims, n_tiles=3, rules=rules, weight_genner=BasicWeightGen([5, 3, 5]), inventory={0: -1, 1: -1}, context_dims=con_dims)

	# Keep stepping till generation is done
	while test.step():  pass
	print(str(test._grid))

	# Use these frequencies in combination with Simple Weight Gen
	rules2, frequencies, mapping = extractRulesAndRelativeFrequencies2D(test._grid)
	print(len(rules))
	print(len(rules2))
	print("Now printing wrong rules")
	l = [rule for rule in rules2 if rule not in rules]; l.sort()
	for r in l:
		print(r)

	print("Now printing ommitted rules")
	l = [rule for rule in rules if rule not in rules2]; l.sort()
	for r in l:
		print(r)

	print("Adjacent tile frequencies")
	for freq in frequencies:
		print(freq)

	test2 = WFCollapse2D(dims=dims, n_tiles=int(len(frequencies)/4), rules = rules2, weight_genner=SimpleWeightGen(frequencies, context_dimensions=con_dims), context_dims=con_dims)
	while test2.step():  pass
	print(str(test2._grid))

	test = WFCollapse2D(dims=dims, n_tiles=3, rules=rules, weight_genner=SimpleWeightGen(frequencies, context_dimensions=con_dims), inventory={0: -1, 1: -1}, context_dims=con_dims)
	while test.step(): pass
	print(str(test._grid))
	print("WaveFunc Program terminated")