import numpy as np
import gym

class SkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(SkipEnv, self).__init__(env)
        self._skip = skip

    def step(self, action):
        t_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = truncated or terminated
            t_reward += reward
            if done:
                break
            
        return obs, t_reward, done, info
    
class PreProcessFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(PreProcessFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, 
                                                high=255,
                                                shape=(80,80,1), 
                                                dtype=np.uint8)
    
    def observation(self, obs):
        return PreProcessFrame.process(obs)
    
    @staticmethod
    def process(frame):
        try:
            new_frame = np.reshape(frame, frame.shape).astype(np.float32)
        except:
            new_frame = np.reshape(frame[0], frame[0].shape).astype(np.float32)
        #convert to grey scale  
        new_frame = 0.299*new_frame[:,:,0] + \
                    0.587*new_frame[:,:,1] + \
                    0.114*new_frame[:,:,2]
        new_frame = new_frame[35:195:2, ::2].reshape(80,80,1)        
        return new_frame.astype(np.uint8)
    
class MoveImgChannel(gym.ObservationWrapper):
    def __init__(self, env):
        super(MoveImgChannel, self).__init__(env)
        #convolution layers are set to be "channel-first"
        #channels are moved to the 1st dimension
        self.observation_space = gym.spaces.Box(low=0.0, high =1.0, 
                                    shape=(self.observation_space.shape[-1],
                                           self.observation_space.shape[0],
                                           self.observation_space.shape[1]),
                                    dtype=np.float32)
        
    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)
    
class ScaleFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0
    
class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        #4 steps are skiped and each observation have 3 channels
        self.observation_space = gym.spaces.Box(
            env.observation_space.low.repeat(n_steps, axis = 0),
            env.observation_space.high.repeat(n_steps, axis = 0),
            dtype = np.float32
        )
        
    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=np.float32)
        return self.observation(self.env.reset())
    
    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        
        return self.buffer
    
def make_env(env_name, is_render = False, mode = "rgb_array"):
    if is_render:
        env = gym.make(env_name, render_mode = mode)
    else:
        env = gym.make(env_name)
    env = SkipEnv(env)
    env = PreProcessFrame(env)
    env = MoveImgChannel(env)
    env = BufferWrapper(env, 4)
    
    return ScaleFrame(env)