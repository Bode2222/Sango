import numpy as np


# A class of object that generates probabilities of how likely a cell is to choose a certain tile based on the context of that cell
class WeightGenerator:
	def gen_weights(self, context, pos):
		pass

# Tile weights independent of the types of cells around it
class BasicWeightGen(WeightGenerator):
	def __init__(self, weights):
		self._weights = weights

	def gen_weights(self, context, pos):
		return self._weights

# Tile weights depend on the tiles chosen to the n, e, w, s of this tile
class SimpleWeightGen(WeightGenerator):
	def __init__(self, weights, context_dimensions):
		# Make the weights a normal vector
		norm = np.linalg.norm(weights)
		self._weights = weights/norm
		self._con_dims = context_dimensions
	
	# context passed in as a 1d array
	def gen_weights(self, context, pos):
		# Average the frequencies of the 4 cells around it to get its frequency. if it doesnt have access to any, return an even distr over all options
		this_index = int(len(context)/2)
		# initialize result to 0 in the length of the number of tiles
		result = np.array([0.0 for x in range(len(self._weights[0]))])

		# if this is neg dont add results
		n = this_index - self._con_dims[1]
		# if this % width == 0 dont add results
		e = this_index + 1
		# if this is >= len(context) dont add results
		s = this_index + self._con_dims[1]
		# if this % width == width - 1 dont add results
		w = this_index - 1
		# normalize even distribution vector
		even_distr = np.array([1/len(self._weights[0]) for x in range(len(self._weights[0]))])
		norm = np.linalg.norm(even_distr)
		even_distr = even_distr/norm

		# if there is a chosen cell there, let it influence this one, otherwise add an evenly distr probability vector
		if n >= 0 and context[n].chosen_tile != -1:
			result += np.array(self._weights[context[n].chosen_tile])
		else:
			result += even_distr
		if e % self._con_dims[1] != 0 and context[e].chosen_tile != -1:
			result += np.array(self._weights[context[e].chosen_tile])
		else:
			result += even_distr
		if s < len(context) and context[s].chosen_tile != -1:
			result += np.array(self._weights[context[s].chosen_tile])
		else:
			result += even_distr
		if w % self._con_dims[1] == self._con_dims[1] - 1 and context[w].chosen_tile != -1:
			result += np.array(self._weights[context[w].chosen_tile])
		else:
			result += even_distr
		
		return result
