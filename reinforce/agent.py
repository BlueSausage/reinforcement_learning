import torch
import torch.optim as optim
from torch.distributions import Categorical
from model import PolicyNetwork

use_cude = torch.cuda.is_available()
device = torch.device("cuda" if use_cude else "cpu")

class Agent(object):
    def __init__(self, n_states, n_actions, hidden_size, learning_rate=0.0001, gamma=0.99) -> None:
        """Reinforce agent that interacts with the environment.
        
        Args:
            n_states (int): Number of states.
            n_actions (int): Number of actions.
            hidden_size (int): Size of hidden layers.
            learning_rate (float): Learning rate.
            gamma (float): Discount factor.
        """

        self.policy = PolicyNetwork(n_states, n_actions, hidden_size).to(device)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        self.gamma = gamma

    def get_action(self, state):
        """Returns actions for given state as per current policy.
        
        Args:
            state: Current state 2D-tensor of shape (n, input_size).
            
        Returns:
            int: Chosen action.
        """
        
        probs = self.policy(state.float()) # Action probabilities
        dist = Categorical(probs)
        action = dist.sample()
            
        return action.item(), dist.log_prob(action)
    
    def learn(self, rewards, log_probs):
        """Update policy using given batch of rewards and log probabilities.
        
        Args:
            rewards (list): List of rewards.
            log_probs (list): List of log probabilities.
        """
        
        returns = []
        G = 0
        for r in rewards[::-1]:
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()