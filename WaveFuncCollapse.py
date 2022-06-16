# Implementation Details here: https://robertheaton.com/2018/12/17/wavefunction-collapse-algorithm/
import math
import random
import numpy as np
from WFUtil import Rule, Dir
from Tuple import Tuple, Tuple2D
from WeightGenerator import BasicWeightGen, SimpleWeightGen
from GenFromImg import extractRulesAndRelativeFrequencies2D


# WF Collapse interface
# to change kind overwrite getTupleObject, propagate, _gen_adjacency matrix
class WaveFunctionCollapse:
	# inventory maps tile id to number of units available for use. if number of units is -1, that means infinite available
	def __init__(self, dims, n_tiles, rules, tile_selection="RANDOM", weight_genner=None, inventory={-1: -1}, context_space= "LOCAL", context_dims=[3, 3], prices={-1: 0}, money=10):
		self._dims = dims
		self._rules = rules
		self._n_tiles = n_tiles
		self._context_dims = context_dims
		self._context_space = context_space
		self._bank = money
		self._orig_bank = money
		self._tile_sel = tile_selection
		# init grid
		self._grid = self._getTupleObject(n_tiles, dims, context_dims, context_space)
		self._adj = self._gen_adjacency_matrix()

		# Weight entropy storage stuff
		self._min_ent_list = set()
		self._min_entropy = float('inf')
		# Buffer to store weight, entropy pairs for each cell
		self._weight_entropy = [[[], float('inf')] for i in range(len(self._grid))]

		# Weight handling. if no weights, give even weights for all
		self._weight_genner = weight_genner
		if weight_genner == None:
			self._weight_genner = BasicWeightGen([1/n_tiles for i in range(n_tiles)])
		
		# Inventory handling. if not set, set all to infinity.
		self._orig_inv = inventory
		if -1 in inventory.keys():
			self._orig_inv = [-1 for i in range(n_tiles)]
		else:
			self._orig_inv = [inventory[i] if i in inventory.keys() else -1 for i in range(n_tiles)]
		self._inv = self._orig_inv

		# Price handling. If not set, set all to 0
		if -1 in prices.keys():
			self._prices = [0 for i in range(n_tiles)]
		else:
			self._prices = [prices[i] if i in prices.keys() else 0 for i in range(n_tiles)]
		# Used to determine when it is time to update every cell. This is when our current currency crosses a tile price
		self._sorted_prices = self._prices.copy()
		self._sorted_prices.sort()
		self._ready_to_buy = np.array([False for i in range(n_tiles)])
		# Move the price tracker so I can know when it crosses a tile piece
		self._set_price_tracker()
		self._price_change = False
		
		# set all the weights in the grid
		self._set_weights([i for i in range(len(self._grid))])

	# Place a cell at a given location.
	def place(self, loc, chosen_tile):
		# Set chosen tile to cell value
		self._grid.get_cell(loc).chosen_tile = chosen_tile
		# Set cell entropy to infinite
		self._weight_entropy[self._grid.loc_to_index(loc)][1] = float('inf')
		# Remove tile from inventory?
		"""if chosen_tile > 0 and self._inv[chosen_tile] > 0:
			self._inv[chosen_tile] -= 1"""
		self._propagate(loc)

	def step(self):
		# Get the location of the cell with the lowest entropy
		loc = self._get_lowest_entropy()
		if (loc[0] == -1): return 0
		# Choose a tile for that cell
		self._collapse(loc)
		# Change what tiles are available for the cell around that tile based on the tile chosen
		self._propagate(loc)
		return 1

	# Reset entire grid to pre changed form
	def reset(self):
		self._grid = self._getTupleObject(self._n_tiles, self._dims, self._context_dims, self._context_space)
		self._inv = self._orig_inv
		self._bank = self._orig_bank
		self._set_price_tracker()
		# Weight entropy storage stuff
		self._min_entropy = float('inf')
		self._min_ent_list = set()
		# Buffer to store weight, entropy pairs for each cell
		self._weight_entropy = [[[], float('inf')] for i in range(len(self._grid))]
		# set all the weights in the grid
		self._set_weights([i for i in range(len(self._grid))])

	# Move the price tracker so I can know when it crosses a tile piece
	# Also update the ready to buy list so i know what I can buy
	def _set_price_tracker(self):
		# update tracker
		self._price_tracker = -1
		for price in self._sorted_prices: 
			if self._bank >= price:
				self._price_tracker += 1

		# update ready to buy list
		self._ready_to_buy = np.array([False if price > self._bank else True for price in self._prices])

	# Calculate all the weight, entropy pairs for all the cells in a list of cell indexes
	# shannon_entropy_for_square = log(sum(weight)) - (sum(weight * log(weight)) / sum(weight))
	def _set_weights(self, loc):
		# If there is nothing to update, set the min_entropy to infinity and call it a day
		if len(loc) == 0:
			self._min_entropy = float('inf')
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
		curCell.tile_active = np.logical_and(curCell.tile_active, self._ready_to_buy)

		# Go through all the active tiles that can be chosen and add up their weights
		for ii in range(len(curCell.tile_active)):
			if curCell.tile_active[ii]:
				# When a tile has run out of stock in the inventory remove it from all cells
				if self._inv[ii] == 0:
					curCell.tile_active[ii] = False
					continue
				no_active_tiles = False
				# To prevent log error if the weight is 0 dont add it
				if weight[ii] != 0:
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
		# If we couldnt calcuate the entropy of a single cell then all the cells have chosen tiles and the program is over
		if self._min_entropy == float('inf'):
			return [-1]
		
		# chosen cell is randomly selected from list of cells at the minimum entropy and removed from list
		if len(self._min_ent_list) == 0:
			print(self._min_entropy)
		
		min_loc = []
		if self._tile_sel == "LINEAR":
			min_loc = self._min_ent_list.pop(0)
		else:
			min_loc = self._min_ent_list.pop(random.randrange(len(self._min_ent_list)))

		# Return the location of the cell with min entropy
		return min_loc

	# Randomly select an option from remaining options
	def _collapse(self, loc):
		# Get list of available tiles
		available_tile_ids = [x for x in range(len(self._grid.get_cell(loc).tile_active)) if self._grid.get_cell(loc).tile_active[x]]
		available_weights = [self._weight_entropy[self._grid.loc_to_index(loc)][0][x] for x in range(len(self._grid.get_cell(loc).tile_active)) if self._grid.get_cell(loc).tile_active[x]]

		if len(available_tile_ids) == 0:
			self._grid.get_cell(loc).chosen_tile = -2
			return -2

		# choose a random tile
		chosen_tile = random.choices(available_tile_ids, weights=available_weights)[0]
		# Set chosen tile to cell value
		self._grid.get_cell(loc).chosen_tile = chosen_tile
		# Set cell entropy to infinite
		self._weight_entropy[loc[0] * self._grid.wid + loc[1]][1] = float('inf')
		# Remove tile from inventory
		if self._inv[chosen_tile] > 0:
			self._inv[chosen_tile] -= 1
		# Subtract price from bank
		self._bank -= self._prices[chosen_tile]
		# Check if bank crossed tile price. if it did, update entire grid and set the new price tracker
		if self._price_tracker >= 0:
			if self._bank < self._sorted_prices[self._price_tracker]:
				self._price_change = True
				self._set_price_tracker()
			elif self._price_tracker < self._n_tiles - 1 and self._bank >= self._sorted_prices[self._price_tracker + 1]:
				self._price_change = True
				self._set_price_tracker()
		else:
			if self._bank >= self._sorted_prices[0]:
				self._price_change = True
				self._set_price_tracker()

		return chosen_tile
		
	# Generate weights given context
	def _gen_weights(self, context, pos):
		return self._weight_genner.gen_weights(context, pos)

	# the cell at location loc has been changed, How does that affect the cells around that one?
	def _propagate(self, loc):
		pass

	# What kind of tensor are we dealing with in this wavefront collapse? 1d, 2d or 3d?
	def _getTupleObject(n_tiles, dims, context_dims) -> Tuple:
		pass

	# For every tile generate a bool list corresponding to every tile that can be beside it in every direction
	def _gen_adjacency_matrix(self):
		pass



# 2D impl of WF Collapse interface
class WFCollapse2D(WaveFunctionCollapse):
	NUM_DIRS = 4
		
	# Sets the tuple object to correct subclass of tuple
	def _getTupleObject(self, n_tiles, dims, context_dims, context_space):
		return Tuple2D(n_tiles, dims, context_dims, context_space=context_space)

	def _gen_adjacency_matrix(self):
		res = [[False for ii in range(self._n_tiles)] for i in range(self._n_tiles * self.NUM_DIRS)]
		for r in self._rules:
			res[r.t1 * self.NUM_DIRS + ((r.dir+2) % self.NUM_DIRS)][r.t2] = True
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