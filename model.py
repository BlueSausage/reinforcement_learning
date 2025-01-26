import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size) -> None:
        """Deep Q-Network (DQN) model.
        
        Args:
            state_size (int): Size of each state.
            action_size (int): Size of each action.
            hidden_size (int): Size of hidden layers.
        """
        
        super(QNetwork, self).__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU()
        )
        
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self.output_layer = nn.Linear(hidden_size, action_size)
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the Q-values for the given state.
        
        Args:
            x (torch.Tensor): State 2D-tensor of shape (n, input_size).
            
        Returns:
            torch.Tensor: Q-values, 2D-tensor of shape (n, action_size).
        """
        
        x = self.input_layer(x)
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.output_layer(x)
        
        return x