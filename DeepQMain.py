import os
import time
import copy
import random
import numpy as np
import tensorflow as tf
from DeepQ import DeepQ
from MathLangEnv import MathLangDeepQEnvAdapter

if __name__ == '__main__':
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
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
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
    net_path = os.path.join(cur_path, "net_saves\\")

    state = np.array([env.process_state(env.reset())])
    print("Before training: " + str(model(state).numpy()))

    # ep decay = 0.0007
    trainer = DeepQ(model, learning_rate=0.05, policy_clone_period=100, epsilon_decay=0.05, batch_size=32)
    #start_time = time.time()
    trainer.train(500, env, steps_per_save=2500, policy_net_save_file=str(net_path), 
                   reward_save_file=str(rew_path), moving_reward_save_file=str(mov_path))
    #print("Time taken: " + str(time.time() - start_time))]
    #model.fit(np.array([[0, 1]]), np.array([[1, 1, 1]]), epochs=100, verbose=0)


    print("input state: " + str(state))
    print("after training: " + str(model(state).numpy()))
    print("after training: " + str(model(np.array([[0.333333333, 0, 0, -0.3333333333]])).numpy()))
    #print("after training: " + str(model(np.array([[0., -0.3333333]])).numpy()))

    print("Deep Q Main Program Finished Execution")