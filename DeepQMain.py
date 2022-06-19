import os
import time
import copy
from typing import final
import numpy as np
import tensorflow as tf
from DeepQ import DeepQ
from MathLangEnv import MathLangDeepQEnvAdapter


if __name__ == '__main__':
    env = MathLangDeepQEnvAdapter()
    input_size = len(env.process_state(env.reset()))
    output_size = env.get_num_actions()

    # Design model
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_size,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(output_size, activation='relu')
    ])
    orig_weights = copy.deepcopy(model.get_weights())

    # Save files
    # Get path to current file
    cur_path = os.path.dirname(__file__)
    rew_path = os.path.join(cur_path, "reward.csv")
    mov_path = os.path.join(cur_path, "mov_rew.csv")
    net_path = os.path.join(cur_path, "net_saves\\")
    # if you wanted to go in the parent dir, append to cur path a seperator and the 'parent_directory' symbol
    path = os.path.normpath(cur_path + os.sep + os.pardir)

    trainer = DeepQ(model, policy_clone_period=100, epsilon_decay=0.0007)
    start_time = time.time()
    trainer.train(5, env, steps_per_save=2500, policy_net_save_file=str(net_path), 
                    reward_save_file=str(rew_path), moving_reward_save_file=str(mov_path))
    print("Time taken: " + str(time.time() - start_time))

    final_weights = model.get_weights()

    state = env.process_state(env.reset())
    print("Orig: " + str(state))
    state = np.array(state)
    state = state.reshape(1, state.shape[0])

    if ((orig_weights == final_weights).all()): 
        print("Hahaha")
    print("Output: " + str(model(state).numpy()))
    print("Model orig weights: " + str(orig_weights))
    print("Final weights: " + str(final_weights))
    print("Deep Q Main Program Finished Execution")