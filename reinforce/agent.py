import torch
import torch.optim as optim
import random
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from model import PolicyNetwork
from torch.distributions import Categorical

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

class REINFORCEAgent():
    def __init__(self, env, hidden_size, engine_failure, learning_rate=0.0001, gamma=0.99, print_every = 100, num_episodes=5000) -> None:
        """Reinforce agent that interacts with the environment.
        
        Args:
            env: The environment to interact with.
            hidden_size (list): List of hidden layer sizes.
            learning_rate (float): Learning rate.
            gamma (float): Discount factor.
            print_every (int): Print every n episodes.
        """
        plt.style.use("seaborn-v0_8-paper")
        self.env = env
        
        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.threshold = env.spec.reward_threshold
        
        self.policy = PolicyNetwork(self.n_states, self.n_actions, hidden_size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.2, total_iters=num_episodes
        )
        
        self.engine_failure = engine_failure
        
        self.gamma = gamma
        self.print_every = print_every
        self.num_episodes = num_episodes
        
        self.no_engine = 0
        self.main_engine = 2
        self.scores = []
        self.avg_scores = []
    

    def get_action(self, state):
        """Returns actions for given state as per current policy.
        
        Args:
            state: Current state 2D-tensor of shape (n, input_size).
            
        Returns:
            int: Chosen action.
        """
        self.policy.eval()
        probs = self.policy(state.float()).squeeze(0) # Action probabilities
        dist = Categorical(probs)
        action = dist.sample()
            
        return action, dist.log_prob(action)
    
    
    def run_episode(self):
        """Run a single episode of the environment with the agent. Return the rewards and log probabilities of the actions taken.
        
        Returns:
            rewards: A list of rewards received at each time step.
            log_probs: A list of log probabilities of the actions taken at each time step.
        """
        state = self.env.reset()[0] # Reset environment and get initial state
        done = False
        log_probs = []
        rewards = []
        
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action, log_prob = self.get_action(state_tensor)
            
            if self.engine_failure == 'random_failure':
                if random.random() < 0.2:
                    action = torch.tensor([[self.no_engine]], device=device)
            elif self.engine_failure == 'main_engine_failure':
                if action.item() == self.main_engine:
                    if random.random() < 0.2:
                        action = torch.tensor([[self.no_engine]], device=device)
            
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            log_probs.append(log_prob)
            rewards.append(reward)
            
            done = terminated or truncated
            
            if done:
                break
            
            state = next_state
            
        return rewards, log_probs
    
    
    def optimize(self, rewards, log_probs):
        """Update policy using given batch of rewards and log probabilities from one episode.
        
        Args:
            rewards (list): List of rewards.
            log_probs (list): List of log probabilities.
        """
        self.policy.train()
        
        # Calculate Monte-Carlo total returns
        returns = torch.zeros(len(rewards), device=device)
        G = 0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            returns[t] = G

        # Calculate baseline and policy loss
        returns = (returns - returns.mean()) / (returns.std().clamp(min=1e-10))
        log_probs = torch.stack(log_probs)
        policy_loss = -1 * (returns * log_probs).sum()
        
        # Optimize the model
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
    
    
    def learn(self, ignore_threshold=False):
        """Train the agent to interact with the environment.
        
        Args:
            num_episodes (int): Number of episodes to train the agent.
            
        Returns:
            scores: List of scores received at each episode.
            avg_scores: List of average scores received at each episode.
        """
        cumulative_score = 0
        time_start = time.time()
        
        for episode in range(self.num_episodes):
            rewards, log_probs = self.run_episode()
            self.optimize(rewards, log_probs)
            
            score = sum(rewards)
            self.scores.append(score)
            
            if len(self.scores) == 1:
                cumulative_score = score
            else:
                cumulative_score += score
                if len(self.scores) > self.print_every:
                    cumulative_score -= self.scores[-self.print_every - 1]
                    
            avg_score = cumulative_score / min(len(self.scores), self.print_every)
            self.avg_scores.append(avg_score)
            
            if episode % self.print_every == 0 or episode == self.num_episodes - 1:
                dt = int(time.time() - time_start)
                time_start = time.time()
                print(f"Episode {episode} - Score: {score} - Avg Score: {avg_score:.2f} - Learning Rate: {self.scheduler.get_last_lr()} - Time: {dt}s")
                
            if avg_score >= self.threshold and not ignore_threshold:
                print(f"Environment solved in {episode} episodes with an average score of {avg_score:.2f}!")
                break
        
        return self.scores, self.avg_scores
    
    
    def plot_scores(self):
        """Plot the scores and average scores of the training progress.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(np.arange(1, len(self.scores) + 1), self.scores, label="Score", color="blue", alpha=0.5, s=1.5)
        ax.plot(np.arange(1, len(self.avg_scores) + 1), self.avg_scores, label="Average Score", color="red", linewidth=2)
        ax.axhline(self.threshold, color="green", label="Threshold")
        
        ax.set_title("Training Progress", fontsize=16)
        ax.set_xlabel("Episode #", fontsize=14)
        ax.set_ylabel("Score", fontsize=14)
        
        ax.legend(loc="best", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)
        
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        
        plt.tight_layout()
        plt.show()
        
    
    def save_model(self, file_name):
        """Save the model with file extension .pth.
        
        Args:
            file_name (str): Name of the file to save the model.
        """
        path = os.path.join(os.getcwd(), "models", file_name)
        torch.save(self.policy.state_dict(), path)
        
        
    def load_model(self, file_name):
        """Load a model with file extension .pth.
        
        Args:
            file_name (str): Name of the file to load the model.
        """
        path = os.path.join(os.getcwd(), "models", file_name)
        self.policy.load_state_dict(torch.load(path, map_location=device))