import random
import numpy as np
from Env import Env, EnvDeepQAdapter
from WFCollapse1DBackContext import WFCollapse1DBackContext
from WFCollapseEnv import WFState, WFAction


class MathLangState(WFState):
	# prompt is the initial unoptimized input ranomly generated at the start of the game
	def __init__(self, prompt, context, loc) -> None:
		self._prompt = prompt
		super().__init__(context, loc)

	def get(self):
		return [self._prompt, self._tuple, self._loc]


# Math lang is a game where you have to place tiles (representing math transformations such as + x or * x) to mimic a 
# given equation also represented as tiles. a list of tiles will be referred to as a program
class MathLangEnv(Env):
	# Max board len
	LEN = 5
	# context length
	CONTEXT_LEN = 3
	# How many random samples to pick when comparing 2 programs
	NUM_SAMPLES = 10
	# What portion of the graph to sample (the combined math functions form a graph of input to output)
	MIN_CAP = -3
	MAX_CAP = 3
	NUM_TILES = 3
	# To change tiles increase num_tiles above, edit cost in init, and add what the tile represents in compile
	# 2 tiles are:
	# 0: + 3, cost: 1
	# 1: - 2, cost: 1
	# 2: + 0, cost: 0
	def __init__(self) -> None:
		self.game = WFCollapse1DBackContext([self.LEN], self.NUM_TILES, context_dims=[self.CONTEXT_LEN])
		self.prompt = self._gen_prompt()
		# How much each tile costs to have in your output
		self._costs = [1, 1, 0]

	# At every step do a wf collapse step then combine with the unoptimized prompt program to get a mathlang state
	# In production the NeuralNet weightgen will have a way to set the prompt so the wfcollapse only has to give the context
	def step(self, action: WFAction):
		state, running = self.game.env_step(action)
		context, loc = state.get()
		return [MathLangState(self.prompt, context, loc), self.reward(), running]

	# Get a new prompt then reset the game
	def reset(self):
		self.prompt = self._gen_prompt()
		state = self.game.reset()
		context, loc = state.get()
		return MathLangState(self.prompt, context, loc)
	
	# compare board state to unoptimzed version(self.prompt) and reward accordingly
	# for weight gen purposed the q value of a state cant be less than 0, so positive rewards only pls
	def reward(self):
		# generated result
		genned = []
		# if board is incomplete, automatic 0 grade
		for cell, _ in self.game._grid:
			if cell.chosen_tile < 0:
				return 0
			genned.append(cell.chosen_tile)
		
		similarity_score = self.compare(genned, self.prompt)
		optimization_score = self.judge_efficiency(genned)
		return similarity_score * (1 + optimization_score)

	def get_num_actions(self):
		return self.NUM_TILES

	# returns a num between 0 and 1 that describes their similarity as relates to their outputs when sampled btw min_cap, max_cap
	def compare(self, genned, prompt):
		# sample btw min_cap and max_cap
		samples = np.random.uniform(self.MIN_CAP, self.MAX_CAP, [self.NUM_SAMPLES])
		diff = 0

		# for every sample, calc prompt result, calc genned result, add the difference
		for sample in samples:
			generated_result = self.compile(genned, sample)
			actual_result = self.compile(prompt, sample)
			diff += abs(generated_result - actual_result)

		# Run the difference through finishing function
		return 1/(diff + 1)

	# given a list of tiles representing math functions and an input, do those math funcitons on that input
	def compile(self, program, input):
		result = input

		for t in program:
			if t == 0:
				result += 3
			elif t == 1:
				result -= 2
			elif t == 2:
				continue

		return result

	def judge_efficiency(self, genned):
		costs = [self._costs[x] for x in genned]
		total = sum(costs)
		efficiency = 1/(abs(total) + 1)
		return efficiency

	def _gen_prompt(self):
		return [random.randrange(self.NUM_TILES) for x in range(self.LEN)]

class MathLangDeepQEnvAdapter(MathLangEnv, EnvDeepQAdapter):
	# returns an action object based on a given state and weight distr
	def make_action(self, state, weights):
		loc = state.get()[2]
		# Get list of available tiles
		available_tile_ids = [x for x in range(len(self.game._grid.get_cell(loc).tile_active)) if self.game._grid.get_cell(loc).tile_active[x]]
		available_weights = [weights[x] for x in range(len(self.game._grid.get_cell(loc).tile_active)) if self.game._grid.get_cell(loc).tile_active[x]]

		# if this happens, something has gone catastrophically wrong
		if len(available_tile_ids) == 0:
			self.game._grid.get_cell(loc).chosen_tile = -2
			print("If this prints, something is terribly wrong. MathLangEnv.py")
			return -2
			#return WFAction(state.get()[2], -2)

		# choose a random tile
		chosen_tile = random.choices(available_tile_ids, weights=available_weights)[0]
		return WFAction(state.get()[2], chosen_tile)

	# Process a given state into a list of values
	def process_state(self, state):
		prompt, context, _ = state.get()

		con_list = [cell.chosen_tile for cell in context]
		return prompt + con_list

if __name__ == '__main__':
	env = MathLangEnv()

	running = True
	reward = 0
	state = env.reset()

	while running:
		state, reward, running = env.step(WFAction(state.get()[2], random.randrange(0, 2)))
		

	print("Environment testing finished")