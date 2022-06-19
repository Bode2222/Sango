# Algorith inspired by DeepLizard Deep q series on youtube video 14 timestep 5:48
import math
import copy
import random
import numpy as np
import tensorflow as tf
from Env import EnvDeepQAdapter


# A class to aid with deep q learning
class DeepQ:
	# Set up the system
	# batch_size = 256
	# gamma = 0.999
	# replay mem size 10k
	# lr = 0.001
	# num_eps = 1k
	def __init__(self, policy_net, steps_per_episode=-1, batch_size=256, replay_mem_size=10000, policy_clone_period=50):
		# Set up replay memory capacity
		self._replay_mem = []
		self._replay_mem_size = replay_mem_size

		# set policy net with random weights
		self._policy_net = policy_net

		# other net init stuff
		self._loss_fn = tf.keras.losses.MeanSquaredError()
		self._policy_net.compile(optimizer='adam', loss=self._loss_fn, metrics=['accuracy'])

		# Clone policy net into target net
		self._target_net = self._clone_policy()

		# init steps per episode
		self._steps_per_episode = steps_per_episode
		if steps_per_episode == -1:
			self._steps_per_episode = float('inf')

		# init exploration threshold
		self._epsilon_thresh = 0
		self._ep_start = 1
		self._ep_end = 0.01
		self._ep_decay = 0.0003

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
	def _decay_epsilon(self, cur_step):
		self._epsilon_thresh = self._ep_end + (self._ep_start - self._ep_end) * \
			math.exp(-1. * cur_step * self._ep_decay)

	# Save the policy network into file
	def _save_policy_net(self, filename: str):
		self._policy_net.save(filename)

	# copy policy network into target net
	def _clone_policy(self):
		self._target_net = tf.keras.models.clone_model(self._policy_net)
		self._target_net.compile(optimizer='adam', loss=self._loss_fn, metrics=['accuracy'])
		self._target_net.set_weights(self._policy_net.get_weights())
		return self._target_net


	# Train the network
	def train(self, n_epi: int, env: EnvDeepQAdapter, steps_per_save=500, policy_net_save_file="", reward_save_file="", moving_reward_save_file=""):
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
			while (running and steps < self._steps_per_episode):
				# Select an action via exploration or exploitation
				epsilon = np.random.uniform(0, 1)
				action_weights = []

				# this is determeined every step cuz we need em for back prop. explanation near 'next_output' variable
				# weights determined by the network. Make state into tensor and pass into network
				net_input = np.array(env.process_state(state))
				net_input = net_input.reshape(1, net_input.shape[0])
				# batch size by input shape
				net_output = self._policy_net(net_input).numpy()
				# with batch size 1 just get the first output
				net_output = net_output[0]
				# add small num to everything cuz weights cant sum to 0
				net_output = [x + 0.00000000000001 for x in net_output]

				if epsilon < self._epsilon_thresh:
					# action weights have even freqeuncy
					action_weights = [1 for x in range(env.get_num_actions())]
				else:
					action_weights = net_output
				# make the chosen action an action object
				action = env.make_action(state, action_weights)

				# execute action in environment (env.step(action)). Store state, reward, running tuple
				n_state, reward, running = env.step(action)

				# store reward in per step reward list
				rewards_per_step.append(reward)

				# store above tuple in replay memory
				self._insert_replay((env.process_state(state), net_output, action, reward, env.process_state(n_state)))

				# Progress state
				state = n_state

				# Sample random batch from replay mem
				batch = random.sample(self._replay_mem, min(len(self._replay_mem), self._batch_size))
				# pull up action weights, action taken, reward and next states
				cur_states = []; a_ws = []; actions = []; rewards = []; n_states = []
				for replay in batch:
					cur_states.append(replay[0])
					a_ws.append(replay[1])
					actions.append(replay[2])
					rewards.append(replay[3])
					n_states.append(replay[4])

				# process all states and next state pairs into workable forms for the network
				n_states = np.array(n_states)
				cur_states = np.array(cur_states)
				rewards = np.array(rewards)

				# Calculate target output (reward + output of target net)
				# pass the batch of states into policy nets and target nets to get the q vals of current and next states
				# for every output in this batch: if this output index corresponds to the action we chose, replace the target val
				# at this point with the true target val (reward + max(target_net(next state))). However! if this is not that action
				# replace this target val with the output of the network so the loss for that node will be 0. All this cuz keras 
				# refuses to employ a backprop function smh
				target = self._policy_net(n_states).numpy()
				target = np.amax(target, axis=1)
				target = [[rewards[ii] + target[ii] if i == actions[ii].get_index() else a_ws[ii][i] for i in range(env.get_num_actions())] for ii in range(len(target))]
				target = np.array(target)

				# Update policy net using gradient descent on loss
				self._policy_net.fit(cur_states, target, verbose=0)

				# after a number of timesteps, copy policy weights into target net once more
				if total_steps % self._clone_period == 0:
					self._clone_policy()

				# Decay exploration rate
				self._decay_epsilon(total_steps)

				# update the number of steps taken so far
				steps += 1
				total_steps += 1

				# if the number of steps % steps_per_save == 0, and we have a save file, save the neural network
				if total_steps % steps_per_save == 0:
					if policy_net_save_file != "":
						self._save_policy_net(policy_net_save_file + "_" + str(int(total_steps/steps_per_save)))

			# sum per step reward list and put it in per episode reward list
			last_reward = sum(rewards_per_step)
			rewards_per_episode.append(last_reward)
			print(str(total_steps) + ": epsilon: " + str(self._epsilon_thresh) + ", Reward: " + str(last_reward))

			# append last episode reward to reward csv file
			if reward_save_file != "":
				reward_file = open(reward_save_file, 'a')
				reward_file.write(", " + str(last_reward))
		
		# calc moving average
		moving_ave = []
		window_size = 200
		i = 0
		
		while i < len(rewards_per_episode) - window_size + 1:
			window = rewards_per_episode[i : i + window_size]
			window_average = round(sum(window) / window_size, 2)
			# append last episode reward to reward csv file
			if moving_reward_save_file != "":
				reward_file = open(moving_reward_save_file, 'a')
				reward_file.write(", " + str(window_average))
			i += 1

		# Save the network
		if policy_net_save_file != "":
			self._save_policy_net(policy_net_save_file + "_" + str(total_steps/steps_per_save))
		return rewards_per_episode
	

if __name__ == '__main__':
	print("Deep q Testing Finished")