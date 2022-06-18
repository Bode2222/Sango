class Env:
    # Returns a state, reward, status list
    def step(self):
        pass

    # Returns a state
    def reset(self):
        pass

    # Returns the number of possible actions in this environment
    def get_num_actions(self):
        pass

class EnvDeepQAdapter(Env):
    # returns an action object based on a given state and weight distr
    def make_action(self, state, weights):
        pass

    # Process a given state into a list of values
    def process_state(self, state):
        pass