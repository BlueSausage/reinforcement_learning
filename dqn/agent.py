import torch
import torch.optim as optim
import random
from model import QNetwork
from replay_buffer import ReplayBuffer, Trajectory

use_cude = torch.cuda.is_available()
device = torch.device("cuda" if use_cude else "cpu")

class Agent(object):
    def __init__(self, n_states, n_actions, hidden_size, batch_size, learning_rate=0.0001) -> None:
        """Deep Q-Network (DQN) agent that interacts with the environment.
        
        Args:
            n_states (int): Number of states.
            n_actions (int): Number of actions.
            hidden_size (int): Size of hidden layers.
            learning_rate (float): Learning rate.
        """
        
        self.model = QNetwork(n_states, n_actions, hidden_size).to(device)
        
        self.mse_loss = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.n_states = n_states
        self.n_actions = n_actions
        
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(10000)
        
        
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
        
    def learn(self, gamma):
        """Update value parameters using given batch of experience tuples.
        
        Args:
            gamma (float): Discount factor.
        """
        
        if self.replay_buffer.__len__() < self.batch_size:
            return
        
        trajectories = self.replay_buffer.sample(self.batch_size)
        
        batch = Trajectory(*zip(*trajectories))
        
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward).view(-1, 1)
        next_states = torch.cat(batch.next_state)
        dones = torch.cat(batch.done).view(-1, 1)

        q_values = self.model(states).gather(1, actions)
        
        max_next_q_values = self.model(next_states).max(1)[0].view(-1, 1)
        target_q_values = rewards + (gamma * max_next_q_values * (1 - dones))
        
        loss = self.mse_loss(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()