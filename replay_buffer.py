import random

class ReplayBuffer(object):
    def __init__(self, capacity) -> None:
        """Fixed-size buffer to store experience tuples."""
        
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
        
    def push(self, batch) -> None:
        """Add a new experience to memory."""
        
        self.memory.append(batch)
        if len(self.memory) > self.capacity:
            del self.memory[0]
        
        
    def sample(self, batch_size) -> list:
        """Randomly sample a batch of experiences from memory."""
        
        return random.sample(self.memory, batch_size)
    
    
    def __len__(self) -> int:
        """Return the current size of internal memory."""
        
        return len(self.memory)