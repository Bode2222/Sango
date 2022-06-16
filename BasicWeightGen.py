from WeightGenerator import WeightGenerator

# Tile weights independent of the types of cells around it
class BasicWeightGen(WeightGenerator):
	def __init__(self, weights):
		self._weights = weights

	def gen_weights(self, context, pos):
		return self._weights