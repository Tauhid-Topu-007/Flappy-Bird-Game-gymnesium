from pickletools import optimize
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
import argparse
    
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
            target_dqn=DQN(num_states,num_actions).to(device)
            #copy the weights from policy_dqn to target_dqn
            target_dqn.load_state_dict(policy_dqn.state_dict())

            steps=0
            self.optimizer=optim.Adam(policy_dqn.parameters(), lr=self.alpha)
            
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
                    
                    steps+=1
                state=new_state
                episode_reward+=reward
            print(f"Episode {episode} with total reward: {episode_reward} and epsilon: {epsilon}")

            if is_training:
                #epsilon decay
                epsilon=max(self.epsilon_min, epsilon * self.epsilon_decay)
            
            if is_training and len(memory)>self.mini_batch_size:
                #get_sample
                mini_batch=memory.sample(self.mini_batch_size)
                
                optimize(mini_batch, policy_dqn, target_dqn)
                
                #sync the target network with policy network
                if steps>self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    steps=0
    # env.close()
    
    def optimize(self, mini_batch, policy_dqn, target_dqn):
    # Unpack batch of experiences
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        # Convert to tensors
        states = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device)
        next_states = torch.stack(next_states).to(device)
        rewards = torch.stack(rewards).to(device)
        terminations = torch.tensor(terminations).float().to(device)

        # -----------------------
        #  Calculate target Q-values
        # -----------------------
        with torch.no_grad():
            # max Q-value for next states from target network
            next_q_values = target_dqn(next_states).max(dim=1)[0]

            # Bellman equation: Q_target = r + γ * max(Q_next) * (1 - done)
            target_q = rewards + (1 - terminations) * self.gamma * next_q_values

        # -----------------------
        #  Q-values from current policy
        # -----------------------
        # Get predicted Q-values for actions taken
        current_q = policy_dqn(states).gather(
            dim=1,
            index=actions.unsqueeze(1)
        ).squeeze(1)

        # -----------------------
        #  Compute loss
        # -----------------------
        loss = self.loss_fn(current_q, target_q)

        # -----------------------
        #  Optimize model
        # -----------------------
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
if __name__=="__main__":
    # Parse command line inputs
    parser = argparse.ArgumentParser(description="Train or test model.")

    parser.add_argument(
        "hyperparameters",
        help=""
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Enable training mode. If not set, testing mode is used."
    )

    args = parser.parse_args()

    # Create DQL agent
    dql = Agent(param_set=args.hyperparameters)

    # Run according to mode
    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)