import pickle
import numpy as np
import copy as copy

EMPTY  = 0 #empty grid
AGENT1 = 1 #players or other agent
AGENT2 = 2 #RL agent

class Agent():
    def __init__(self, gamma = 0.99, n_grid = 9, n_label = 3, epsilon = 1e-5):
        
        self.gamma    = gamma
        self.epsilon  = epsilon
        self.n_grid   = n_grid
        self.n_label  = n_label

        self.V = dict()
        self.Q = dict()
        self.policy = dict()

        #stores all the possible states
        #state = 9 * 1 vector = combinations of empty grid + agent1's action + agent2's action 
        self.states = []

    def intialise(self):

        self.gen_all_states()

        for s in self.states:
            # if self.who_win(s) is not None: 
            #     self.V[s] = self.get_reward(s)
            # else:
            self.V[s] =  0

            self.Q[s] = [0]*self.n_grid

    def is_double_winner(self, s):

        '''
        0 1 2
        3 4 5
        6 7 8
        '''

        h_cnt = v_cnt = 0
        for i in range(3):
            # check horizontal lines
            if s[i*3] == s[i*3 + 1] and s[i*3] == s[i*3 + 2] and s[i*3] > 0:
                h_cnt += 1

            #check vertical lines
            if s[i] == s[i + 3] and s[i] == s[i + 6] and s[i] > 0:
                v_cnt += 1

        if h_cnt >= 2 or v_cnt >= 2:
            return True
        else:
            return False
    
    def is_invalid_state(self, s, who_first):

        if who_first == AGENT1:
            #if agent1 first => 1,2,1,... => when in the non-terminal state => agent2's turn => agent1's move always more than agent2 by "1"
            if s.count(AGENT1) - s.count(AGENT2) != 1:
                return True
        else:
            #if agent2 first => 2,1,..    => when in the non-terminal state => agent2's turn => agent2's move always equal to agent1 
            if s.count(AGENT1) != s.count(AGENT2):
                return True

        return False
    
    def gen_all_states(self):

        #generate all possible states
        for s in np.ndindex(tuple([3]*self.n_grid)):

            #check if double winners => impossible
            if self.is_double_winner(s):
                continue

            self.states.append(tuple(s))

    def who_win(self, s):

        '''
        0 1 2
        3 4 5
        6 7 8
        '''

        #check vertical and horizontal line
        for i in range(3):
            if s[i*3] == s[i*3 + 1] and s[i*3] == s[i*3 + 2] and s[i*3] > 0:
                return s[i*3]
            if s[i] == s[i + 3] and s[i] == s[i + 6] and s[i] > 0:
                return s[i]
        
        #check diagonal
        if s[0] == s[4] and s[0] == s[8] and s[0] > 0:
            return s[0]
        if s[2] == s[4] and s[2] == s[6] and s[2] > 0:
            return s[2]
        
        return None
        
    def get_reward(self, s):
        
        winner = self.who_win(s)
        if winner is None:
            return 0
        else:
            return 1 if winner == AGENT2 else -1

    def gen_next_actions_states(self, agent, s):

        actions = []
        next_states = []

        for i in range(len(s)):
            if s[i] == EMPTY:
                actions.append(i)
                new_state = list(copy.copy(s)) #for modifying data
                new_state[i] = agent
                next_states.append(tuple(new_state)) #for hashinig 

        return actions, next_states

    def get_transition_p(self, s, actions):
        #self.get_transition_p(ns, actions_agent1) is always 1 and return 0 in terminal state
        #since only one new state and one reward could be obtained based on current state and specific action taken
        #in tic-tac-toe game
        if self.who_win(s) is not None:
            return 0
        
        return 1.
    
    def extract_policy(self):   
        
        for s in list(self.Q.keys()):
            if self.who_win(s) is None:
                max_value = -np.inf
                max_index = None
                for i in range(self.n_grid):
                    if s[i] == EMPTY and max_value < self.Q[s][i]:
                        max_value = self.Q[s][i]
                        max_index = i

                self.policy[tuple(s)] = max_index

    def save_policy(self):
        with open('policy.pkl', 'wb') as f:
            pickle.dump(self.policy, f)
    
    def load_policy(self):
        with open('policy.pkl', 'rb') as f:
            self.policy = pickle.load(f)

    def iterate(self, who_first):

        self.intialise()

        i = 0
        while True:
            delta = 0.
            for cs in self.states:
                
                #ensure the current state is AGENT2's turn
                if self.is_invalid_state(cs, who_first):
                    continue

                #check if current state is the terminal state => no next move => skip
                if self.who_win(cs) is not None: #is not None = someone wins
                    continue

                old_v, self.Q[cs] = self.V[cs], [0]*self.n_grid

                #generate all possible actions and next states of agent 2
                actions_agent2, next_states_agent2 = self.gen_next_actions_states(AGENT2, cs)  

                for a2, ns2 in zip(actions_agent2, next_states_agent2):
                    
                    #update self.V[ns2] when the next move of agent 2 have not won yet
                    if self.who_win(ns2) != AGENT2:
                        #update self.V[ns2] based on agent1's possible actions and next states
                        self.V[ns2] = 0
                        actions_agent1, next_states_agent1 = self.gen_next_actions_states(AGENT1, ns2)  
                        for a1, ns1 in zip(actions_agent1, next_states_agent1):
                            #here, I assume the policy of agent 1 is uniformly distributed
                            self.V[ns2] += 1/len(actions_agent1)*self.get_transition_p(ns2, actions_agent1)*(self.get_reward(ns1) + self.gamma*self.V[ns1])

                    #update self.Q[cs][a2]
                    self.Q[cs][a2] = self.get_transition_p(cs, actions_agent2)*(self.get_reward(ns2) + self.gamma*self.V[ns2])

                self.V[cs] = max(self.Q[cs])
                delta = max(delta, abs(self.V[cs] - old_v))

            print(f"iteration {i} - delta: {delta}")
            if delta < self.epsilon:
                break

            i += 1

        self.extract_policy()
        self.save_policy()

    def step(self, s):

        return self.policy[s]