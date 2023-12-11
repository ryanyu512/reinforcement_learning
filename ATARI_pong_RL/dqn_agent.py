from keras.layers import Activation, Dense, Conv2D, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras import backend as K
import numpy as np

from utils import *
from env_create import *

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape):
        #define max memory size
        self.mem_size = max_size 
        
        #memory counter
        self.mem_cntr = 0 
        
        #initialise state memory
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                      dtype = np.float32)
        #initialise new state memory
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), 
                                         dtype = np.float32)
        #initialise action memory
        self.action_memory = np.zeros(self.mem_size, dtype = np.int32)
        
        #initialise reward memory
        self.reward_memory = np.zeros(self.mem_size, dtype = np.float32)
        
        #initialise terminal memory (is it done?)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)
        
    def store_transition(self, state, action, reward, state_, done):
        
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        
        self.mem_cntr += 1
        
    def sample_buffer(self, batch_size):
        #randamly sample memory in the size of batch_size 
        #replace = False => index cannot be chosen multiple times
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace = False)
        
        #extract batch of memory
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        
        return states, actions, rewards, new_states, dones        

def build_dqn(lr, n_actions, input_dims, fcl_dims):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size = 8, strides = 4, activation = 'relu',
                        input_shape = (*input_dims,), data_format='channels_first'))
    model.add(Conv2D(filters=64, kernel_size = 4, strides=2, activation= 'relu',
                        data_format='channels_first'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu',
                        data_format='channels_first'))
    model.add(Flatten())
    model.add(Dense(fcl_dims, activation = 'relu'))
    model.add(Dense(n_actions))
    
    model.compile(optimizer=Adam(lr=lr), loss='mean_squared_error')
    
    return model
    
class Agent(object):
    def __init__(self, alpha, gamma, n_actions, n_windows, epsilon, batch_size, replace,
                    input_dims, eps_dec=1e-5, eps_min=0.01, mem_size=1000000,
                    q_eval_fname='q_eval.h5', q_target_fname='q_target.h5'):
        
        #initialise action space
        self.action_space = [i for i in range(n_actions)]
        #initialise discount factor
        self.gamma = gamma
        #define threshold for taking random actions
        self.epsilon = epsilon
        #define amount of decremental value of epsilon value
        self.eps_dec = eps_dec
        #define minimum epsilon value
        self.eps_min = eps_min
        #define batch training size
        self.batch_size = batch_size
        #define when the q_next network is replaced
        self.replace = replace
        #define q_next file name
        self.q_target_model_file = q_target_fname
        #define q_eval file name
        self.q_eval_model_file = q_eval_fname
        #initialise learning step
        self.learn_step = 0
        #initialise sliding windows for computing average scores
        self.n_windows = n_windows
        #define memory 
        self.memory = ReplayBuffer(mem_size, input_dims)
        #define q_eval network
        self.q_eval = build_dqn(alpha, n_actions, input_dims, fcl_dims = 512)
        #define q_next network
        self.q_next = build_dqn(alpha, n_actions, input_dims, fcl_dims = 512)
        
    def replace_target_network(self):
        #if learn_step % self.replace == 0 => weights of q_next = weights of q_eval
        if self.replace != 0 and self.learn_step % self.replace == 0:
            self.q_next.set_weights(self.q_eval.get_weights())
            
    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        
    def choose_action(self, observation):
        #self.epsilon is decreased gradually after one learning step
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation], copy=False, dtype=np.float32)
            actions = self.q_eval.predict(state, verbose = 0)
            action = np.argmax(actions)
            
        return action

    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            #extact a set of learning information
            state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
            
            #update q_next network when condition is met
            self.replace_target_network()
            
            #predict the state_action value based on current state
            q_eval = self.q_eval.predict(state, verbose = 0)
            #predict the state_action value based on next state
            q_next = self.q_next.predict(new_state, verbose = 0)
            #set state_action value based on next state = 0 when the next state = done
            q_next[done] = 0.0
            
            indices = np.arange(self.batch_size)
            q_target = q_eval[:]
        
            '''
            if done => no update on q_target since q_next[done] = 0.0
            if not done => update on q_target based on reward + self.gamma*np.max(q_next, axis = 1)
            '''
            q_target[indices, action] = reward + self.gamma*np.max(q_next, axis = 1)
            
            self.q_eval.train_on_batch(state, q_target)
            
            if self.epsilon > self.eps_min:
                self.epsilon = self.epsilon - self.eps_dec 
            else:
                self.epsilon = self.eps_min
                
            self.learn_step += 1
    
    def train(self, num_games = 250):
        #define enviornment
        env = make_env('PongNoFrameskip-v4', is_render = False)

        #initialise best score 
        best_score = -21

        #initialise scores and epsilon history
        scores, eps_history = [], []
        
        #initilsie number of step
        n_steps = 0
        
        #start learning
        for i in range(num_games):
            score = 0
            observation = env.reset()
            done = False
            
            while not done:
                #choose action
                action = self.choose_action(observation)
                #interact with the environment
                observation_, reward, done, info = env.step(action)
                
                #update n_step and total score
                n_steps += 1
                score += reward

                #store transition 
                self.store_transition(observation, action, reward, observation_, int(done))
                
                #learn 
                self.learn()
                
                #update observation
                observation = observation_
            
            #update scores history
            scores.append(score)
            
            #get average scores
            avg_score = np.mean(scores[-self.n_windows:])
            print('episode ', i, 'score', score, 'average score %.2f' % avg_score, 'epsilon %.2f' % self.epsilon, 'steps', n_steps)
            
            #save model if avg_scre 
            if avg_score > best_score and i >= self.n_windows:
                self.save_models()
                print('avg score %.2f better than best score %.2f' % (avg_score, best_score))
                best_score = avg_score
            
            #update epsilon history
            eps_history.append(self.epsilon)
        
        #plot and save scores history 
        x = [i + 1 for i in range(num_games)]
        filename = 'PongNoFrameSkip-v4.png'
        plotLearning(x, scores, eps_history, filename)
        
        #close environment
        env.close()

    def demo(self, is_save_gif = False, mode = "human"):
        #define render mode
        if is_save_gif:
            mode = "rgb_array"
        #define enviornment
        env = make_env('PongNoFrameskip-v4', is_render = True, mode = mode)
        #initialise agent
        #change epsilon from 1.0 to 0.02 to ensure less random actions are chosen
        self.epsilon = 0.02
        #load model
        self.load_models()
        
        #start playing
        observation = env.reset()
        done = False
        frames = []
        while not done:
            action = self.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            observation = observation_
            if is_save_gif:
                frames.append(env.render())
            else:
                env.render()
        
        if is_save_gif:
            save_frames_as_gif(frames)
        
        #close environment
        env.close()
        
    def save_models(self):
        self.q_eval.save(self.q_eval_model_file)
        self.q_next.save(self.q_target_model_file)
        print("... saving model ...")
        
    def load_models(self):
        self.q_eval = load_model(self.q_eval_model_file)
        self.q_next = load_model(self.q_target_model_file)
        print('... loading models ...')