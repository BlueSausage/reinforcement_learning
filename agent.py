import torch
import torch.optim as optim
import random
from model import QNetwork

use_cude = torch.cuda.is_available()
device = torch.device("cuda" if use_cude else "cpu")

class Agent(object):
    def __init__(self, n_states, n_actions, hidden_size, learning_rate=0.0001) -> None:
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
        
    def learn(self, state, action, reward, next_state, done, gamma):
        """Update value parameters using given batch of experience tuples.
        
        Args:
            state (torch.Tensor): Current state.
            action (torch.Tensor): Action taken in the state.
            reward (torch.Tensor): Reward received after taking action.
            next_state (torch.Tensor): Next state.
            done (torch.Tensor): Whether the episode is complete or not.
            gamma (float): Discount factor.
        """

        q_values = self.model(state).gather(1, action)
        
        max_next_q_values = self.model(next_state).max(1)[0].view(-1, 1)
        target_q_values = reward + (gamma * max_next_q_values * (1 - done))
            
        loss = self.mse_loss(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()