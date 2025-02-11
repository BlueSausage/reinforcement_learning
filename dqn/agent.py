import torch
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from model import QNetwork
from replay_buffer import ReplayBuffer, Trajectory

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

class DQNAgent():
    def __init__(self, env, hidden_size, batch_size=64, replay_buffer_size=10000, learning_rate=0.0001, gamma=0.99, min_epsilon=0.01, max_eps_episode=150, num_episodes=1000, print_every=100) -> None:
        """Deep Q-Network (DQN) agent that interacts with the environment.
        
        Args:
            env: The environment to interact with.
            hidden_size (int): Size of hidden layers.
            batch_size (int): Size of the batch to sample from the replay buffer.
            learning_rate (float): Learning rate.
            gamma (float): Discount factor.
            min_epsilon (float): Minimum epsilon value.
            max_eps_episode (int): Maximum number of episodes to decay epsilon.
            num_episodes (int): Number of episodes to train the agent.
            print_every (int): Print every n episodes.
        """
        plt.style.use("seaborn-v0_8-darkgrid")
        self.env = env
        
        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.threshold = env.spec.reward_threshold

        self.model = QNetwork(self.n_states, self.n_actions, hidden_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.mse_loss = torch.nn.MSELoss()
        
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        
        self.gamma = gamma
        self.min_epsilon = min_epsilon
        self.max_eps_episodes = max_eps_episode
        self.max_num_episodes = num_episodes
        self.print_every = print_every
        
        self.scores = []
        self.avg_scores = []

    
    def exponential_epsilon_decay(self, episode: int) -> float:
        """Exponential decrease in 𝜖. Initially, 𝜖 falls rapidly and then asymptotically approaches min_epsilon.
    
        Args:
            episode (int): The current episode number.
            
        Returns:
            float: The 𝜖 value for the current episode.
        """
        return self.min_epsilon + (1.0 - self.min_epsilon) * np.exp(-1.0 * episode / self.max_eps_episodes)
    
    
    def linear_epsilon_decay(self, episode):
        """Linear decrease in 𝜖, whereby the value decreases evenly as the episode increases.
            
        Args:
            episode (int): The current episode number.
                
        Returns:
            float: The 𝜖 value for the current episode.
        """
        return max(self.min_epsilon, 1.0 - (episode / self.max_eps_episodes))

    
    def draw_epsilon_decay(self, epsilon_decay=exponential_epsilon_decay):
        """Draws the decay of epsilon over the course of the training episodes.
        
        Args:
            epsilon_decay (function): The epsilon decay function.
        """
        epsilon_values = [epsilon_decay(ep) for ep in range(self.max_eps_episodes)]

        plt.plot(epsilon_values)
        plt.axhline(y=self.min_epsilon, color="red", linestyle="--", label="min_epsilon")
        plt.title("Exponential Decay of Epsilon")
        plt.xlabel("Episode #")
        plt.ylabel("Epsilon")
        plt.show()
        
        
    def get_action(self, state, eps, explorate=True):
        """Returns actions for given state as per current policy.
        
        Args:
            state: Current state 2D-tensor of shape (n, input_size).
            eps (float): Epsilon, for epsilon-greedy action selection (exploration).
            check_eps (bool): If False, no epsilon check is performed.
            
        Returns:
            int: Chosen action.
        """
        sample = random.random()
        
        if not explorate or sample > eps:
            with torch.no_grad():
                action = self.model(state.float()).argmax(dim=1).view(1, 1)
        else:
            action = torch.tensor([[random.randrange(self.n_actions)]], device=device)
        
        return action
    
    
    def run_episode(self, eps):
        """Run a single episode of the environment and train the agent.
        
        Args:
            env (gym.Env): The environment to run.
            eps (float): The epsilon value to use for the episode.
            
        Returns:
            float: The total reward for the episode.
        """
        state = self.env.reset()[0]
        done = False
        total_reward = 0
        
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = self.get_action(state_tensor, eps)
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            total_reward += reward
            
            done = terminated or truncated
            
            if done:
                reward = -1
                
            self.replay_buffer.push((
                state, 
                action,
                reward,
                next_state, 
                done
            ))
            
            self.optimize()
                
            state = next_state
            
        return total_reward
        
        
    def optimize(self):
        """Update value parameters using given batch of experience tuples.
        """
        
        if self.replay_buffer.__len__() < self.batch_size:
            return
        
        trajectories = self.replay_buffer.sample(self.batch_size)
        
        batch = Trajectory(*zip(*trajectories))
        
        states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=device)
        actions = torch.tensor(batch.action, dtype=torch.int64, device=device).view(-1, 1)
        rewards = torch.tensor(np.array(batch.reward), dtype=torch.float32, device=device).view(-1, 1)
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=device)
        dones = torch.tensor(np.array(batch.done), dtype=torch.int64, device=device).view(-1, 1)

        q_values = self.model(states).gather(1, actions)
        max_next_q_values = self.model(next_states).max(1)[0].view(-1, 1)
        target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)
        
        loss = self.mse_loss(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
    def learn(self, epsilon_decay_fn, ignore_threshold=True):
        """Train the agent to interact with the environment.
        
        Args:
            epsilon_decay_fn (function): The epsilon decay function to use.
            ignore_threshold (bool): If True, the training will continue until max_num_episodes is reached.
        
        Returns:
            scores: List of scores received at each episode.
            avg_scores: List of average scores received at each episode.
        """
        cumulative_score = 0
         
        time_start = time.time()
        
        for episode in range(self.max_num_episodes):
            eps = epsilon_decay_fn(episode)
            score = self.run_episode(eps)
            
            self.scores.append(score)
            
            if len(self.scores) == 1:
                cumulative_score = score
            else:
                cumulative_score += score
                if len(self.scores) > self.print_every:
                    cumulative_score -= self.scores[-self.print_every - 1]
            
            avg_score = cumulative_score / min(len(self.scores), self.print_every)
            self.avg_scores.append(avg_score)
            
            if episode % self.print_every == 0 or episode == self.max_num_episodes-1:
                dt = int(time.time() - time_start)
                time_start = time.time()
                print(f"Episode {episode} - Score: {score} - Avg Score: {avg_score:.2f} - Epsilon: {eps:.4f} - Time: {dt}s")
                
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
        ax.axhline(self.threshold, color="green", linestyle="--", label="Threshold")
        
        ax.set_title("Training Progress", fontsize=16)
        ax.set_xlabel("Episode #", fontsize=14)
        ax.set_ylabel("Score", fontsize=14)
        
        ax.legend(loc="best", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)
        
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        
        plt.tight_layout()
        plt.show()
        
        
    def save(self, file_name):
        """Save the model with file extension .pth.
        
        Args:
            file_name (str): Name of the file to save the model.
        """
        path = os.path.join(os.getcwd(), "models", file_name)
        torch.save(self.model.state_dict(), path)
        
        
    def load_model(self, file_name):
        """Load a model with file extension .pth.
        
        Args:
            file_name (str): Name of the file to load the model.
        """
        path = os.path.join(os.getcwd(), "models", file_name)
        self.model.load_state_dict(torch.load(path, map_location=device))