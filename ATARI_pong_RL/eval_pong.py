from dqn_agent import Agent

if __name__ == '__main__':
    agent = Agent(gamma = 0.99,
            epsilon = 0.02, 
            alpha = 0.0001, 
            input_dims = (4,80,80), 
            n_actions = 6, 
            n_windows = 100,
            mem_size = 25000, 
            eps_min=0.02, 
            batch_size=32, 
            replace = 1000, 
            eps_dec = 1e-5)
    
    agent.demo(is_save_gif=True)