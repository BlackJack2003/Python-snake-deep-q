import snake_realist as snake
import show
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sys import argv
import time,random
import signal,os
import pickle

stime=time.time()
strike_l=3
strike=0
# Configuration paramaters for the whole setup
seed = 42
gamma = 0.99  # init at .99 try reducing Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.01  # Minimum epsilon greedy parameter
epsilon_max = 0.8  # Maximum epsilon greedy parameter
epsilon_interval = (epsilon_max - epsilon_min)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 100
rfc=0
ph=0
fpos = [(2,2),(2,snake.size-2),(snake.size-3,snake.size-3),(snake.size-3,3),(snake.size//2,snake.size//2),(2,2),(3,snake.size-3),(snake.size-3,2),(snake.size-2,snake.size-2),(2,2)]
# Use the Baseline Atari environment because of Deepmind helper functions]
m = fpos.copy() if fpos!=None else None
env = snake.snake_board(fpos=m)
# Warp the frames, grey scale, stake four frame and scale to smaller ratio

num_actions = 4

def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(snake.size,snake.size,2,))
    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 3, strides=1, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)
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
epsilon_random_frames = 2000
# Number of frames for exploration
epsilon_greedy_frames = 8000
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
psize=1
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
slen = 0
timestep =0

snake_size=1

def l_opt():
    global optimizer
    try:
        checkpoint = tf.train.Checkpoint(optim=optimizer)
        checkpoint.restore('./mod3/ckpt')
        '''with open("./qvf.pkl",'rb') as f:
            idk__ = pickle.load(f)
        with open('./sample.pkl','rb') as f:
            m = pickle.load(f)
        optimizer.from_config(idk__)
        optimizer.apply_gradients(m)
        optimizer.set_weights(wts)'''
        print("\nOptimizer loaded\n")
    except Exception as e:
        print("\nOptimizer and model not loaded due to:\n"+str(e))

def l_mod():
    global model
    global model_target
    global epsilon_random_frames
    try:
        a = keras.models.load_model('./mod1f/m1.h5')
        b = keras.models.load_model('./mod2f/m2.h5')
        epsilon_random_frames/=5
        print("\nLoaded Models Succesfully\n")
        msnk=1
        pmsnk = 1
        mtot=1
        model=a
        model_target=b
    except Exception as e:
        print("\nModel not loaded due to:\n"+str(e))

size_to_win = 16 if fpos==None else len(fpos)-3 

def eval_mod(k:list=None):
    global fpos
    input("Show ?:")
    _= []
    state = env.reset(fpos=fpos.copy())
    for i in range(max_steps_per_episode//2):
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = model(state_tensor, training=False)
        # Take best action
        action = tf.argmax(action_probs[0]).numpy()
        state_next, reward, done, snake_size = env.step(action)
        _.append(action)
        state_next = np.array(state_next)
        state_next = state
        if done:
            break
    env.render(_,fpos=fpos.copy())
    k = input("Conitnue(Y/N):")
    if k.lower()!="y":
        quit()
    else:
        global size_to_win
        size_to_win = max(size_to_win+1,len(fpos))

def save_t():
    model.save("./mod1f/m1.h5")
    model_target.save("./mod2f/m2.h5")
    checkpoint = tf.train.Checkpoint(optim=optimizer)
    checkpoint.save('./mod3/ckpt')
    '''with open('./optp.pkl', 'wb') as f:
        pickle.dump(optimizer,f)'''
    with open("./qvf.pkl",'wb') as f:
        pickle.dump(optimizer.get_config(),f)
    try:
        with open('./sample.pkl', 'wb') as f:
            pickle.dump(zip(grads, model.trainable_variables),f)
    except:
        pass

csh = 1

if "-rop" in argv:
    optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
    print("\nLoading new optimizer...\n")
else:
    l_opt()
if "-rm" in argv:
    model = create_q_model()
    model_target = create_q_model()
    epsilon_random_frames*=5
    print("\nLoading new neural network...\n")
else:
    l_mod()
if "-rand" in argv:
    nfpos = []
    for m in range(25):
        nfpos.append((random.randint(2,snake.size-2),random.randint(2,snake.size-2)))
    fpos=nfpos
if "-trand" in argv:
    fpos=None
else:
    print("Use -rm for new neural network -rop for new optimizer -rand for random locations...")

size_to_win = 16 if fpos==None else len(fpos)-3 

def handle_exit(signum,frame):
    res = input("Save(y/n):")
    if res.lower()=="y":
        save_t()
        quit()
    else:
        quit()

signal.signal(signal.SIGINT, handle_exit)
relist = [(0,1),(1,0),(2,3),(3,2)]
tlist = {0:2,1:3,2:0,3:1}
paction = None
while True:
    random.shuffle(fpos)     # Run until solved
    m= fpos.copy() if fpos!=None else None
    state = np.array(env.reset(m))
    episode_reward = 0
    csh = 1
    for timestep in range(1, max_steps_per_episode):
        # env.render(); Adding this line would show the attempts
        # of the agent in a pop up window.
        if frame_count%10000==0:
            print("\nSaving model...\n")
            save_t()
            msnk=1 #max size in save
            seconds = time.time()-stime
            minutes, seconds = divmod(seconds, 60)
            hours, minutes = divmod(minutes, 60)
            print("\nCurrent Run Time:%d:%02d:%02d\n" % (hours, minutes, seconds))                
        frame_count += 1
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            if epsilon>1:
                epsilon-=0.3
            action = np.random.choice(num_actions)
            apa_pair = (action,paction)
            if apa_pair in relist:
                action = tlist[action]
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
        paction = action
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
            updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)
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
            template = "avg rew: {:.2f} at episode {}, frame count {},Num rand frame: {}, reward: {:.2f}, snake size:{}, Timstep: {}, epsilon:{:0.2f}, Deaths: {},current save:{} ,max_size:{}, max num of frame:{}, pszie: {}"
            print(template.format(mrh_, episode_count, frame_count,rfc,reward,snake_size,timestep,epsilon,deaths,msnk,mtot,max_f_d,psize))

        # Limit the state and reward history
        if len(rewards_history) > max_memory_length:
            del rewards_history[:timestep+2]
            del state_history[:timestep+2]
            del state_next_history[:timestep+2]
            del action_history[:timestep+2]
            del done_history[:timestep+2]
        if done:
            deaths+=1
            break
            
    # Update running reward to check condition for solving
    psize=snake_size
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
    if snake_size>size_to_win:  # Condition to consider the task solved
        save_t()
        print("Solved at episode {}! at action number {} with snake size: {}".format(episode_count,timestep,snake_size))
        #eval_mod(action_history[-timestep:])
        with open("./showff.pkl","wb") as f:
            pickle.dump((action_history[-timestep:],fpos),f)
        try:
            eval_mod()
        except:
            show.show()
        #os.system("shutdown /s")