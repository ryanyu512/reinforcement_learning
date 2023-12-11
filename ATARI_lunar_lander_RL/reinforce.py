import gym
import numpy as np
import tensorflow as tf

from utils import plotLearning
import matplotlib.pyplot as plt
from matplotlib import animation

from tensorflow.keras.layers import Dense, Activation, Input 
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam 

class Policy_Network(Model):
    def __init__(self, n_actions, h_dims = [256, 256]):
        super(Policy_Network, self).__init__()
        self.h_dims = h_dims
        self.n_actions = n_actions
        
        self.fc1 = Dense(self.h_dims[0], activation = 'relu')
        self.fc2 = Dense(self.h_dims[1], activation = 'relu')
        self.policy = Dense(n_actions, activation = 'softmax')
    
    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        p = self.policy(x)
        
        return p
    
class Agent(object):
    def __init__(self, 
                 lr = 0.0005, 
                 gamma = 0.99, 
                 n_actions = 4, 
                 n_episodes = 2000,
                 n_windows = 20,
                 layer_size = [256, 256],
                 model_file = 'reinforce'):
        
        # initialise parameters
        self.gamma = gamma
        self.lr = lr
        self.G = 0.
        self.layer_size = layer_size
        self.n_actions  = n_actions
        self.n_episodes = n_episodes
        self.n_windows  = n_windows
        
        # initialise memory 
        self.state_memory  = []
        self.action_memory = []
        self.reward_memory = []
        
        # define policy net
        self.policy_net = Policy_Network(self.n_actions, self.layer_size)
        self.policy_net.compile(optimizer = Adam(learning_rate = self.lr))
        
        # initialise action space
        self.action_space = [i for i in range(n_actions)]
        
        # define model filename
        self.model_file = model_file
    
    def choose_action(self, observation, is_training = False):
        
        # compute policy distribution
        s = tf.convert_to_tensor([observation])
        p = self.policy_net(s, training = is_training)[0]
        
        # normalise policy distribution
        tmp = p
        tmp = tf.cast(tmp, dtype = tf.float64)
        tmp = tmp*(1./tf.reduce_sum(tmp))
        
        # sample actions
        a = np.random.choice(self.action_space, p = tmp)
        
        # compute log probability
        log_p = tf.math.log(p[a])
        
        return a, log_p
    
    def store_transition(self, observation, action, reward):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
        
    def learn(self, log_ps, tape):

        reward_memory = np.array(self.reward_memory, dtype = np.float32)
        
        # compute return 
        G = np.zeros_like(reward_memory)
        for t in range(len(reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k]*discount
                discount *= self.gamma
            
            G[t] = G_sum
        
        # compute loss = compute expected returns
        # J = sum(log(p)*g) => higher = better
        # loss = -sum(log(p)*g) => minimize loss = maximise expected returns
        loss = 0
        for g, log_p in zip(G, log_ps):
            loss += -g * log_p
        
        # compute gradient
        gradient = tape.gradient(loss, self.policy_net.trainable_variables)
        self.policy_net.optimizer.apply_gradients(zip(gradient, self.policy_net.trainable_variables))
        
        # clear memory
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

    def train(self, is_load_model = True):
        
        if is_load_model:
            self.load_model()
        
        env = gym.make('LunarLander-v2')
        score_history = []
        
        windows    = self.n_windows
        n_episode  = self.n_episodes
        best_score = -np.inf
        
        
        for i in range(n_episode):
            done = False
            score = 0
            obs = env.reset()
            
            log_ps = []
            with tf.GradientTape () as tape:
                while not done:
                    action, log_p = self.choose_action(obs)
                    obs_, reward, done, info = env.step(action)
                    self.store_transition(obs, action, reward)
                    obs = obs_
                    score += reward
                    
                    log_ps.append(log_p)
                
                score_history.append(score)
                
                cost = self.learn(log_ps, tape)
        
            score_avg = np.mean(score_history[-windows:])
            
            if score_avg > best_score and i >= windows:
                best_score = score_avg
                self.save_model()
                
            print(f'episode: {i}, score: {score}, avg_score: {score_avg}, best_score: {best_score}')
            
        filename = 'lunar_lander.png'
        x = [i + 1 for i in range(n_episode)]
        plotLearning(x = x, 
                    scores = score_history, 
                    filename = filename, 
                    window = 20)
        
    def demo(self, n_games = 1, is_animate = False):
        
        self.load_model()
        
        env = gym.make('LunarLander-v2')
        
        best_score = -np.inf
        frames = []
        for i in range(n_games):
            done = False
            score = 0
            obs = env.reset()

            while not done:
                action, log_p = self.choose_action(obs)
                obs_, reward, done, info = env.step(action)
                obs = obs_
                score += reward
                if is_animate:
                    frames.append(env.render(mode="rgb_array"))
                else:
                    env.render()
                
            print(f'episode: {i}, score: {score}')
        
        env.close()
        save_frames_as_gif(frames)
        
    def save_model(self):
        self.policy_net.save(self.model_file, save_format = 'tf')
        print("====save model====")
        
    def load_model(self):
        self.policy_net = load_model(self.model_file)
        print("====load model====")

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    
    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)