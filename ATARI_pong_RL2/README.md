# ATARI_pong_RL
![gym_animation](https://github.com/ryanyu512/ATARI_pong_RL/assets/19774686/04c8a0fc-f80b-407b-9b93-026bf4b115de)

It is fancinating to combine computer vision and reinforcement learning concept to train AI agent to play ATARI game - pong. In this project, classical deep Q learning network is adopted to evaluate which action is the most appropriate to gain more scores. 

1. train_pong.py: used to set hyper-parameters and train AI agent
2. eval_pong.py: used to evaluate the trained AI agent
3. dqn_agent.py: define DQN
4. env_create.py: define environment, pre-processing and interaction
5. utils.py: customised plotting functions
6. q_eval.h5 and q_target.h5: trained DQN model data
7. gym_animation.gif: short gif for demonstration
