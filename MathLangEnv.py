import random
import numpy as np
from WFCollapseEnv import State, Action
from WFCollapse1D import WFCollapse1D


class MathLangState(State):
    # prompt is the initial unoptimized input ranomly generated at the start of the game
    def __init__(self, prompt, context, loc) -> None:
        self._prompt = prompt
        super().__init__(context, loc)

    def __init__(self, prompt, other: State) -> None:
        context, loc = other.get()
        self._prompt = prompt
        super().__init__(context, loc)

    def get(self):
        return [self._prompt, self._tuple, self._loc]

class Env:
    def step(self):
        pass

# Math lang is a game where you have to place tiles (representing math transformations such as + x or * x) to mimic a 
# given equation also represented as tiles. a list of tiles will be referred to as a program
class MathLangEnv(Env):
    # Max board len
    LEN = 5
    # How many random samples to pick when comparing 2 programs
    NUM_SAMPLES = 5
    # What portion of the graph to sample (the combined math functions form a graph of input to output)
    MIN_CAP = -3
    MAX_CAP = 3
    NUM_TILES = 2
    # 2 tiles are:
    # 0: + 3, cost: 1
    # 1: - 2, cost: 1
    def __init__(self) -> None:
        self.game = WFCollapse1D([self.LEN], self.NUM_TILES)
        self.prompt = self._gen_prompt()
        # How much each tile costs to have in your output
        self._costs = [1, 1]

    # At every step do a wf collapse step then combine with the unoptimized prompt program to get a mathlang state
    # In production the NeuralNet weightgen will have a way to set the prompt so the wfcollapse only has to give the context
    def step(self, action: Action):
        state, running = self.game.env_step(action)
        return [MathLangState(self.prompt, state), self.reward(), running]

    # Get a new prompt then reset the game
    def reset(self):
        self.prompt = self._gen_prompt()
        return MathLangState(self.prompt, self.game.reset())
    
    # compare board state to unoptimzed version(self.prompt) and reward accordingly
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

        return result

    def judge_efficiency(self, genned):
        costs = [self._costs[x] for x in genned]
        total = sum(costs)
        efficiency = 1/(abs(total) + 1)
        return efficiency

    def _gen_prompt(self):
        return [0]

if __name__ == '__main__':
    env = MathLangEnv()

    running = True
    reward = 0
    state = env.reset()
    while running:
        state, reward, running = env.step(Action(state.get()[2], random.randrange(0, 2)))
        

    print("Environment testing finished")