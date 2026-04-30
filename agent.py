import random

import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import torch
import yaml
import torch.nn as nn
import torch.optim as optim

if torch.backend.mps.is_available():
    device='mps'
elif torch.cuda.is_available():
    device='cuda'
else:
    device='cpu'

class Agent:
    def __init__(self,param_set):
        self.param_set=param_set
        with open('parameters.yaml') as f:
            all_param_set=yaml.safe_load(f)
            params=all_param_set[param_set]
            
        self.alpha=params['alpha']
        self.gamma=params['gamma']
        self.epsilon_init=params['epsilon_init']
        self.epsilon_min=params['epsilon_min']
        self.epsilon_decay=params['epsilon_decay']
        self.replay_memory_size=params['replay_memory_size']
        self.mini_batch_size=params['mini_batch_size']
        self.reward_threshold=params['reward_threshold']
        self.network_sync_rate=params['network_sync_rate']
        
        
        self.loss_fn=nn.MSELoss()
        self.optimizer=None
    def run(self,is_training=True,render=False):

        env = gym.make("FlappyBird-v0", render_mode="human" if render else None) 

        num_states=env.observation_space.shape[0] #input dim
        num_actions=env.action_space.n #output dim

        policy_dqn=DQN(num_states,num_actions).to(device)

    

        if is_training:
            memory=ReplayMemory(self.replay_memory_size)
            epsilon=self.epsilon_init
            
        for episode in itertools.count():

            state, _ = env.reset()
            state=torch.tensor(state, dtype=torch.float).to(device)
            episode_reward=0
            terminated=False

            while not terminated:
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample() 
                    action=torch.tensor(action, dtype=torch.long).to(device)
                    # Explore action space
                else:
                    with torch.no_grad():
                        action = policy_dqn.act(state.unsqueeze(dim=0)).squeeze().argmax()  # Exploit learned values

                # Processing:
                next_state, reward, terminated, _, = env.step(action.item())
                
                #create tensors
                reward=torch.tensor(reward, dtype=torch.float).to(device)
                next_state=torch.tensor(next_state, dtype=torch.float).to(device)
                if is_training:
                # Store the experience in the replay memory
                    memory.append((state, action, new_state,reward, terminated))
                state=new_state
                episode_reward+=reward
            print(f"Episode {episode} with total reward: {episode_reward} and epsilon: {epsilon}")


            #epsilon decay
            epsilon=max(self.epsilon_min, epsilon * self.epsilon_decay)
    # env.close()