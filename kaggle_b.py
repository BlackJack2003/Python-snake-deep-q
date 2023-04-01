import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sys import argv
import time
import pickle
import random
import turtle

size= 40
rf = 10/(2*size -1)

class InvalidInputError(Exception):
    print("Invalid Input val")
    
class player:
    def __init__(self,x=size//2 +1,y=size//2 + 1):
        self.cx = x
        self.cy = y
        self.px =x
        self.py =y

class snake_board:
    def elpepe(self)->tuple:
        m = self.fpos[0]
        self.fpos.pop(0)
        return m

    def pepe(self):
        m,k = random.randint(0,size-1),random.randint(0,size-1)
        while self.board[m][k][0]!=0:
            m,k = random.randint(0,size-1),random.randint(0,size-1)
        return m,k

    def __init__(self,fpos=None):
        self.h = player()
        self.board = np.zeros((size,size,2),dtype=np.int16)
        self.segs = [self.h]
        self.board[self.h.cx][self.h.cy][0]=255
        self.board[self.h.cx][self.h.cy][1]=255
        if fpos==None:
            self.getfrp = lambda:self.pepe() 
        else:
            self.fpos = fpos
            self.getfrp = lambda: self.elpepe()
        self.fx,self.fy = self.getfrp()
        self.board[self.fx][self.fy][1]=255
        self.ps=abs(self.fx-self.h.cx) + abs(self.fy-self.h.cy)
        self.size=1
        self.pd = -1
        self.timestep=0

    def check_death(self)->bool:
        cx = self.h.cx
        cy = self.h.cy
        for m in range(1,len(self.segs)):
            if self.segs[m].cx == cx and self.segs[m].cy == cy:
                return True
        return False
    
    def check_eat(self)->bool:
        m = bool(self.h.cx==self.fx and self.h.cy==self.fy)
        if m==True:
            self.fx,self.fy = self.getfrp()
            self.board[self.fx][self.fy][1]=255
            last = self.segs[-1]
            self.board[last.px][last.py][0]=255
            self.board[last.px][last.py][1]=255
            self.segs.append(player(last.px,last.py))
            self.size+=1
        return m
    
    #0 up,1 down 2 left 3 right
    def move(self,dd:int):
        if dd==0:
            dirx=1
            diry=0
        elif dd==1:
            dirx=-1
            diry=0
        elif dd==2:
            dirx=0
            diry=1
        elif dd==3:
            dirx=0
            diry=-1
        else:
            raise InvalidInputError
        self.h.px=self.h.cx
        self.h.py=self.h.cy
        self.h.cx-=dirx
        self.h.cy-=diry
        if self.h.cx < 0:
            self.h.cx=size-1
        elif self.h.cx > size-1:
            self.h.cx=0
        elif self.h.cy<0:
            self.h.cy=size-1
        elif self.h.cy> size-1:
            self.h.cy=0
        #check for border collision
        #trailing segments occupy the preceeding ones place
        self.board[self.h.cx][self.h.cy][0]=255
        self.board[self.h.cx][self.h.cy][1]=255
        self.board[self.h.px][self.h.py][1]=0
        m=0
        for m in range(1,len(self.segs)):
            self.segs[m].px=self.segs[m].cx
            self.segs[m].py=self.segs[m].cy
            self.segs[m].cx = self.segs[m-1].px
            self.segs[m].cy = self.segs[m-1].py
        #set last ones position as free
        self.board[self.segs[-1].px][self.segs[-1].py][0]=0
    
    def step(self,action:int):
        self.move(action)
        eat = self.check_eat()
        self.timestep+=1
        d = self.check_death() 
        _ =abs(self.fx-self.h.cx) + abs(self.fy-self.h.cy)
        if eat==True:
            rew=50*(self.size-1)
        elif d:
            rew=-40
        else:
            rew= 1 if _ < self.ps else -1
        self.ps = _
        return self.board,rew,d,self.size
    
    def reset(self,fpos:list=None):
        self.h = player()
        self.board = np.zeros((size,size,2),dtype=np.int16)
        m = np.ones(size,dtype=np.int16)
        self.segs = [self.h]
        self.board[self.h.cx][self.h.cy][0]=255
        self.board[self.h.cx][self.h.cy][1]=255
        if fpos==None:
            self.getfrp = lambda:self.pepe()
        else:
            self.fpos=fpos
            self.getfrp=lambda:self.elpepe()
        self.fx,self.fy = self.getfrp()
        self.board[self.fx][self.fy][1]=255
        self.ps=abs(self.fx-self.h.cx) + abs(self.fy-self.h.cy)
        self.size=1
        self.timestep=0
        return self.board
    
    def render(self,actions,fpos):
        k = size*10
        wn = turtle.Screen()
        wn.tracer(0)
        self.reset(fpos)
        wn.title("Snake Game")
        wn.bgcolor("white")
        # the width and height can be put as user's choice
        wn.setup(width=max(500,size*21), height=max(500,size*21))
        head=turtle.Turtle()
        head.penup()
        head.setpos((self.h.cy*20)-k,(-20*self.h.cx)+k)
        head.shape('square')
        head.color('black')
        segs=[head]
        food = turtle.Turtle()
        food.shape('square')
        food.color('blue')
        food.penup()
        food.setpos((self.fy*20)-k,(self.fx*-20)+k)
        def add_seg(x,y):
            seg1 = turtle.Turtle()
            seg1.shape('square')
            seg1.color('black')
            seg1.penup()
            seg1.goto(x,y)
            return seg1
        k_ = len(actions)
        for _ in range(len(actions)):
            self.step(actions[_])
            food.setpos((self.fy*20)-k,(self.fx*-20)+k)
            if len(self.segs)>len(segs):
                segs.append(add_seg((self.segs[-1].cy*20)-k,(self.segs[-1].cx*-20)+k))
            for i,v in enumerate(self.segs):
                segs[i].setpos((v.cy*20)-k,(v.cx*-20)+k)
            print("Remianing:"+str(k_-_)+" Fpos:"+str(self.fy)+","+str(self.fx))
            k_-=1
            time.sleep(0.5)
            wn.update()
        _ = input()
        turtle.bye()
    
    def __str__(self)->str:

        tot = "\n    "
        for i in range(size):
            tot+=' '+str(i)
        tot+='\n     '
        for i in range(size):
            tot+=' #'
        tot+="\n"
        for i in range(size):
            r=str(i)+"# "
            for j in range(size):
                m = self.board[i][j]
                r+=' '
                if m[0]==0:
                    if m[1]==255:
                        r+='2'
                    else:
                        r+='0'
                else:
                    if m[1]==0:
                        r+='#'
                    else:
                        r+='H'
            tot+='\n'+r
        return tot+'\nSize: '+str(self.size)+'#'+str(self.h.cx)+'#'+str(self.h.cy)

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
max_steps_per_episode = 2500
rfc=0
ph=0
fpos = [(1,1),(1,size-2),( size-2,1),( size-2, size-2),( size//2, size//2),(1,1),(1, size-2),( size-1,0),( size-1, size-1),(2,2)]
# Use the Baseline Atari environment because of Deepmind helper functions
env =  snake_board(fpos=fpos)
# Warp the frames, grey scale, stake four frame and scale to smaller ratio

num_actions = 4

def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=( size, size,2,))
    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(16, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(32, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(16, 3, strides=1, activation="relu")(layer2)
    layer4 = layers.Flatten()(layer3)
    layer5 = layers.Dense(256, activation="relu")(layer4)
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
epsilon_random_frames = 10000
# Number of frames for exploration
epsilon_greedy_frames = 20000
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 20000
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
    fp =[(0,0),( size//2, size//2)]
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
    if opl:
        with open('./opt.pkl', 'wb') as f:
            pickle.dump(optimizer.get_weights(),f)

csh = 1

while True:  # Run until solved
    m = fpos.copy()
    state = np.array(env.reset(m))
    episode_reward = 0
    csh = 1
    for timestep in range(1, max_steps_per_episode):
        # env.render(); Adding this line would show the attempts
        # of the agent in a pop up window.
        if frame_count%10000==0:
            if opl==True:
                print("\nSaving model...\n")
                save_t()
            msnk=1 #max size in save
            seconds = time.time()-stime
            minutes, seconds = divmod(seconds, 60)
            hours, minutes = divmod(minutes, 60)
            print("\nCurrent Run Time:%d:%02d:%02d\n" % (hours, minutes, seconds))
            if ol!=2:
                ol+=1
            else:
                if ol==2:
                    print("/n Trying to Load optimizer\n")
                    try:
                        with open('./opt.pkl','rb') as f:
                            wts = pickle.load(f)
                        optimizer.set_weights(wts)
                        print("\nOptimizer loaded\n")
                    except Exception as e:
                        print("\nOptimizer not loaded due to:\n"+str(e))
                    try:
                        a = keras.models.load_model('./mod1/m1.h5')
                        b = keras.models.load_model('./mod2/m2.h5')
                        epsilon_random_frames/=10
                        model=a
                        model_target=b
                        print("\nLoaded Models Succesfully\n")
                    except Exception as e:
                        print('no save found due to:',e)
                    msnk=1
                    pmsnk = 1
                    mtot=1
                    opl=True
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
        if mtot<snake_size:
            reward+=50
            mtot = snake_size
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
            template = "avg rew: {0:.2f} at episode {1}, frame count {2},Num rand frame: {3}, reward: {4:.2f},snake size:{5},epsilon:{6:0.2f},deaths: {7},current save:{8} ,max_size:{9}, max num of frame:{10}"
            print(template.format(mrh_, episode_count, frame_count,rfc,reward,snake_size,epsilon,deaths,msnk,mtot,max_f_d))

        


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
    if snake_size>=len(fpos)-1 if fpos!=None else 5:  # Condition to consider the task solved
        save_t()
        print("Solved at episode {}!".format(episode_count))
        break


