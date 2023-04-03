import snake
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sys import argv
import time
import copy
import signal
import pickle

stime=time.time()
strike_l=3
strike=0
# Configuration paramaters for the whole setup
seed = 42
gamma = 0.99  # init at .99 try reducing Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 0.8  # Maximum epsilon greedy parameter
epsilon_interval = (epsilon_max - epsilon_min)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 1500
rfc=0
ph=0
fpos = [(1,1),(1,snake.size-2),(snake.size-2,snake.size-3),(snake.size-2,2),(snake.size//2,snake.size//2),(1,1),(1,snake.size-2),(snake.size-1,0),(snake.size-1,snake.size-1),(2,2)]
# Use the Baseline Atari environment because of Deepmind helper functions]
m = fpos.copy()
env = snake.snake_board(fpos=m)
# Warp the frames, grey scale, stake four frame and scale to smaller ratio

num_actions = 4

def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(snake.size,snake.size,2,))
    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 1, strides=1, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 1, strides=1, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 1, strides=1, activation="relu")(layer2)
    layer4 = layers.Flatten()(layer3)
    layer5 = layers.Dense(512, activation="relu")(layer4)
    layer6 = layers.Dense(128, activation="relu")(layer5)
    action = layers.Dense(num_actions, activation="linear")(layer6)

    return keras.Model(inputs=inputs, outputs=action)



# The first model makes the predictions for Q-values which are used to
# make a action.
model = create_q_model()
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_target = create_q_model()


# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
deaths = 0
running_reward = 0
episode_count = 0
frame_count = 0
# Number of frames to take random action and observe output
epsilon_random_frames = 1000
# Number of frames for exploration
epsilon_greedy_frames = 5000
# Maximum replay length
# Note: The Deepmind paper suggests 10_00_000 however this causes memory issues
max_memory_length = 10000
# Train the model after 4 actions
update_after_actions = 4
ol = 0
opl = False
# How often to update the target network
update_target_network = 900
# Using huber loss for stability
msnk=1
pmsnk = 1
mtot=1
loss_function = keras.losses.Huber()
pshow =0
max_f_d=0
updated_q_values = []

optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

snake_size=1

def eval_mod():
    fp =[(0,0),(snake.size//2,snake.size//2)]
    _ = env.reset()
    __ = []
    for i in range(100):
        state_tensor = tf.convert_to_tensor(_)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = model(state_tensor, training=False)
        # Take best action
        action = tf.argmax(action_probs[0]).numpy()
        __.append(action)
        _, reward, done, snake_size = env.step(action)
        if done:
            break
    
    env.render(__,fpos=fp)

def save_t():
    model.save("./mod1/m1.h5")
    model_target.save("./mod2/m2.h5")
    with open('./opt.pkl', 'wb') as f:
        m = optimizer.get_weights()
        print(f"Length of weights:{len(m)}")
        pickle.dump(m,f)
    with open("./qvf.pkl",'wb') as f:
        pickle.dump(optimizer.get_config(),f)
    '''with open('otdp.pkl','wb') as f:
        m = (action_history,state_history,state_next_history,rewards_history,done_history,episode_reward_history,deaths,running_reward,episode_count,frame_count)
        pickle.dump(m,f)'''

csh = 1

if len(argv)>1:
    if argv[1]=="-rm":
        opl=True
        print("\nLoading new neural network...\n")
    else:
        print("Use -rm for new neural network")
        quit()

def handle_exit(signum,frame):
    res = input("Save the data points(y/n):")
    if res.lower()=="y":
        save_t()
    quit()

signal.signal(signal.SIGINT, handle_exit)

while True:  # Run until solved
    m = fpos.copy()
    state = np.array(env.reset(m))
    episode_reward = 0
    csh = 1
    for timestep in range(1, max_steps_per_episode):
        # env.render(); Adding this line would show the attempts
        # of the agent in a pop up window.
        if frame_count%10000==0:
            if opl:
                print("\nSaving model...\n")
                save_t()
            msnk=1 #max size in save
            seconds = time.time()-stime
            minutes, seconds = divmod(seconds, 60)
            hours, minutes = divmod(minutes, 60)
            print("\nCurrent Run Time:%d:%02d:%02d\nLength of optimizer weights:%d\n" % (hours, minutes, seconds,len(optimizer.get_weights())))
            if not opl:
                if ol>=2:
                    print("\n Trying to Load optimizer\n")
                    a_model = model
                    b_model = model_target
                    opti__ = optimizer
                    opl=True
                    try:
                        with open('./opt.pkl','rb') as f:
                            wts = pickle.load(f)
                        with open("./qvf.pkl",'rb') as f:
                            idk__ = pickle.load(f)
                        '''with open('otdp.pkl','rb') as f:
                            m = pickle.load(f)
                            action_history,state_history,state_next_history,rewards_history,done_history,episode_reward_history,deaths,running_reward,episode_count,frame_count = m'''
                                                
                        optimizer.set_weights(wts)
                        print("\nOptimizer loaded\n")
                        a = keras.models.load_model('./mod1/m1.h5')
                        b = keras.models.load_model('./mod2/m2.h5')
                        epsilon_random_frames/=10
                        print("\nLoaded Models Succesfully\n")
                        msnk=1
                        pmsnk = 1
                        mtot=1
                        model=a
                        model_target=b
                    except Exception as e:
                        opl=False
                        model = a_model
                        model_target=b_model
                        optimizer = opti__
                        print("\nOptimizer and model not loaded due to:\n"+str(e))
                else:  
                    ol+=1
                
        frame_count += 1
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            if epsilon>1:
                epsilon-=0.3
            action = np.random.choice(num_actions)
            rfc+=1
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()

        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)
        # Apply the sampled action in our environment
        state_next, reward, done, snake_size = env.step(action)
        msnk = max(msnk,snake_size)
        mtot = max(mtot,snake_size)
        max_f_d = max(timestep,max_f_d)
        state_next = np.array(state_next)
        episode_reward += reward
        # Save actions and states in replay buffer
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward)
        state = state_next
        # Update every fourth frame and once batch size is over 32
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(done_history)), size=batch_size)
            # Using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor([float(done_history[i]) for i in indices])
            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = model_target.predict(state_next_sample,verbose=0)
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * tf.reduce_max(
                future_rewards, axis=1)

            # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)
                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if frame_count % update_target_network == 0:
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())
            # Log details
            mrh_ = np.mean(rewards_history)
            template = "avg rew: {:.2f} at episode {}, frame count {},Num rand frame: {}, reward: {:.2f}, snake size:{}, Timstep: {}, epsilon:{:0.2f}, Deaths: {},current save:{} ,max_size:{}, max num of frame:{}"
            print(template.format(mrh_, episode_count, frame_count,rfc,reward,snake_size,timestep,epsilon,deaths,msnk,mtot,max_f_d))

        # Limit the state and reward history
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]
        if done==True:
            deaths+=1
            break
    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)
    if msnk<pmsnk:
            if strike>strike_l:
                strike=0
                epsilon+=0.5
            else:
                strike+=1
    psnk=msnk
    episode_count += 1
    if snake_size>5 if fpos!=None else 5:  # Condition to consider the task solved
        save_t()
        print("Solved at episode {}! at action number {}".format(episode_count,timestep))
        break


