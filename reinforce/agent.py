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
            hidden_size (list): List of hidden layer sizes.
            learning_rate (float): Learning rate.
            gamma (float): Discount factor.
        """
        self.gamma = gamma
        
        self.policy = PolicyNetwork(n_states, n_actions, hidden_size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
    

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
            
        return action.item(), dist.log_prob(action)
    
    def learn(self, rewards, log_probs):
        """Update policy using given batch of rewards and log probabilities.
        
        Args:
            rewards (list): List of rewards.
            log_probs (list): List of log probabilities.
        """
        self.policy.train()
        returns = torch.zeros(len(rewards), device=device)
        G = 0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            returns[t] = G
        
        returns = (returns - returns.mean()) / (returns.std().clamp(min=1e-10))
        log_probs = torch.stack(log_probs)
        policy_loss = -1 * (returns * log_probs).sum()
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()