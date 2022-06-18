# Algorith inspired by DeepLizard Deep q series on youtube video 14 timestep 5:48
import random
import numpy as np
#import tensorflow as tf
from Env import EnvDeepQAdapter
from WFCollapseEnv import Action


# A class to aid with deep q learning
class DeepQ:
	# Set up the system
	def __init__(self, policy_net, steps_per_episode=-1, batch_size=32, replay_mem_size=512, policy_clone_period=64):
		# Set up replay memory capacity
		self._replay_mem = []
		self._replay_mem_size = replay_mem_size

		# init policy net with random weights

		# Clone policy net into target net
		self._copy_policy_target()

		# init steps per episode
		self._steps_per_episode = steps_per_episode
		if steps_per_episode == -1:
			self._steps_per_episode = float('inf')

		# init exploration threshold
		self._epsilon_thresh = 1

		# init the number of replay memories we will be learning in each step of the game
		self._batch_size = batch_size

		# How frequently to copy policy network into target network
		self._clone_period = policy_clone_period

	# stores a replay in replay mem
	def _insert_replay(self, replay):
		self._replay_mem.append(replay)
		while len(self._replay_mem) > self._replay_mem_size:
			self._replay_mem.pop(0)

	# Reduces the exploration/exploitation threshold
	def _decay_epsilon(self):
		pass

	# Copies policy network into target network
	def _copy_policy_target(self):
		pass

	# Save the policy network into file
	def _save_policy_net(self, filename: str):
		pass

	# Train the network
	def train(self, n_epi: int, env: EnvDeepQAdapter, steps_per_save=500, policy_net_save_file="", reward_save_file=""):
		# init a var to count the number of steps taken so far
		total_steps = 0

		# init a list to store training rewards per episode
		rewards_per_episode = []

		# for every episode
		for i in range(n_epi):
			# init the start state of the env
			state = env.reset()
			reward = 0
			running = True

			# init a var to keep track of steps this episode
			steps = 0

			# init a list to store training rewards per timestep
			rewards_per_step = []

			# for each time step
			while(running and steps < self._steps_per_episode):
				# Select an action via exploration or exploitation
				epsilon = np.random.uniform(0, 1)
				action_weights = []
				if epsilon < self._epsilon_thresh:
					# action weights have even freqeuncy
					action_weights = [1 for x in range(env.get_num_actions())]
				else:
					# actions have weights determined by the network
					pass
				# make the chosen action an action object
				action = env.make_action(state, action_weights)

				# execute action in environment (env.step(action)). Store state, reward, running tuple
				n_state, reward, running = env.step(action)

				# store reward in per step reward list
				rewards_per_step.append(reward)

				# store above tuple in replay memory
				self._insert_replay((state, action, reward, n_state))

				# Progress state
				state = n_state

				# Sample random batch from replay mem
				batch = random.sample(self._replay_mem, min(len(self._replay_mem), self._batch_size))
				cur_states = []; actions = []; rewards = []; n_states = []
				for replay in batch:
					cur_states.append(replay[0])
					actions.append(replay[1])
					rewards.append(replay[2])
					n_states.append(replay[3])

				# process all states and next state pairs into workable forms for the network
				cur_states = list(map(env.process_state, cur_states))
				n_states = list(map(env.process_state, n_states))


				# pass the batch of states into policy nets and target nets
				#cur_states = tf.data.Dataset.from_tensor_slices(cur_states)
				#n_states = tf.data.Dataset.from_tensor_slices(n_states)

				#for el in cur_states:
				#	print(el.numpy())

				# Calculate loss btw output (ouptput policy net) and target (reward + output of target net)

				# Update policy net using gradient descent on loss

				# after a number of timesteps, copy policy weights into target net once more
				if total_steps % self._clone_period == 0:
					self._copy_policy_target()

				# update the number of steps taken so far
				steps += 1
				total_steps += 1

				# Decay exploration rate
				self._decay_epsilon()

				# if the number of steps % steps_per_save == 0, and we have a save file, save the neural network
				if total_steps % steps_per_save == 0:
					if policy_net_save_file != "":
						self._save_policy_net(policy_net_save_file + "_" + str(int(total_steps/steps_per_save)))

			# sum per step reward list and put it in per episode reward list
			rewards_per_episode.append(sum(rewards_per_step))

			# append last episode reward to reward csv file
			if reward_save_file != "":
				pass
		
		# Save the network
		self._save_policy_net(policy_net_save_file + "_" + str(int(total_steps/steps_per_save)))
		return rewards_per_episode
	

if __name__ == '__main__':
	print("Deep q Testing Finished")