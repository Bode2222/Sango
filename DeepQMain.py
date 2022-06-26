import os
import time
import random
import numpy as np
import tensorflow as tf
from DeepQ import DeepQ
from MathLangEnv import MathLangDeepQEnvAdapter

if __name__ == '__main__':
    LOAD_MODEL = False
    TRAIN_MODEL = True
    # for reproducible results
    # this changes what prompt the mathlangenv gives
    random.seed(123)
    # these change the network weights
    np.random.seed(1243)
    tf.random.set_seed(12435)

    env = MathLangDeepQEnvAdapter()
    input_size = len(env.process_state(env.reset()))
    output_size = env.get_num_actions()

    # Design model
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_size,)),
        tf.keras.layers.Dense(1280, activation='tanh'),
        tf.keras.layers.Dense(1280, activation='tanh'),
        tf.keras.layers.Dense(1280, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(output_size)
    ])

    # Save files
    # Get path to current file
    cur_path = os.path.dirname(__file__)
    rew_path = os.path.join(cur_path, "reward.csv")
    mov_path = os.path.join(cur_path, "mov_rew.csv")
    net_path = os.path.join(cur_path, "net_saves\\Len" + str(env.LEN))
    load_path = os.path.join(cur_path, "net_saves\\Len" + str(env.LEN) + "_19")

    print(net_path)

    # ep decay = 0.00008, train for 15k
    # to get len 2 to work only need 2 layers ep decay .05
    trainer = DeepQ(model, learning_rate=0.006, policy_clone_period=100, epsilon_decay=0.0005, replay_mem_size=15000, batch_size=128)

    if LOAD_MODEL:
        trainer.load(str(load_path))
    if TRAIN_MODEL:
        start_time = time.time()
        trainer.train(10500, env, steps_per_save=2000, policy_net_save_file=str(net_path), 
                       reward_save_file=str(rew_path), moving_reward_save_file=str(mov_path))
        print("Time taken: " + str(int((time.time() - start_time)/60)) + " mins and " + str(int((time.time() - start_time) % 60)) + " seconds")

    # Play 16 games with model
    trainer.play(16, env)
    print("Deep Q Main Program Finished Execution")