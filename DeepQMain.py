import os
import time
import random
import numpy as np
import tensorflow as tf
from DeepQ import DeepQ
from MathLangEnv import MathLangDeepQEnvAdapter

if __name__ == '__main__':
    TRAIN_MODEL = True
    # for reproducible results
    # this changes what prompt the mathlangenv gives
    random.seed(123)
    # these change the network weights
    np.random.seed(123)
    tf.random.set_seed(123)

    env = MathLangDeepQEnvAdapter()
    input_size = len(env.process_state(env.reset()))
    output_size = env.get_num_actions()

    # Design model
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_size,)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(output_size)
    ])
    loss_fn = tf.keras.losses.MeanSquaredError()
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss=loss_fn)

    # Save files
    # Get path to current file
    cur_path = os.path.dirname(__file__)
    rew_path = os.path.join(cur_path, "reward.csv")
    mov_path = os.path.join(cur_path, "mov_rew.csv")
    net_path = os.path.join(cur_path, "net_saves\\Len3")
    load_path = os.path.join(cur_path, "net_saves\\_3")

    print(net_path)

    state = np.array([env.process_state(env.reset())])
    print("Before training: " + str(model(state).numpy()))

    # ep decay = 0.0007
    # to get len 2 to work only need 2 layers ep decay .05
    trainer = DeepQ(model, learning_rate=0.01, policy_clone_period=100, epsilon_decay=0.0007, batch_size=128)

    if TRAIN_MODEL:
        start_time = time.time()
        trainer.train(4000, env, steps_per_save=2000, policy_net_save_file=str(net_path), 
                       reward_save_file=str(rew_path), moving_reward_save_file=str(mov_path))
        print("Time taken: " + str(time.time() - start_time))
    else:
        trainer.load(str(load_path))

    # Play 5 games with model
    trainer.play(5, env)

    print("input state: " + str(state))
    print("after training: " + str(model(state).numpy()))

    print("Deep Q Main Program Finished Execution")