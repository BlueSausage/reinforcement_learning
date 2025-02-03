## DQN description

### Model Class

The class QNetwork inherits from torch.nn.Module, which is a base class for all neural network modules in PyTorch. This class implements a Deep Q-Network (DQN), a type of neural network used in reinforcement learning to approximate the Q-value function.

The __init__ method initializes the network's architecture. It takes three parameters:

- state_size, which represent the size of the input state
- action_size, the size of the output actions
- hidden_size, and the size of the hidden layers.

The method first calls the parent class's initializer using super(QNetwork, self).__init__(). Then, it defines the network layers using torch.nn.Sequential, which is a container module that sequences the layers together.

The network consists of an input layer, two hidden layers, and an output layer. Each layer is a linear transformation followed by a ReLU activation function, except for the output layer, which is just a linear transformation. The input layer transforms the input state from state_size to hidden_size. The two hidden layers further transform the data within the hidden_size dimensions. Finally, the output layer transforms the data from hidden_size to action_size, producing the Q-values for each action.

The forward method defines the forward pass of the network. It takes a tensor x as input, representing the state, and sequentially passes it through the input layer, the two hidden layers, and the output layer. The method returns the resulting tensor, which contains the Q-values for the given state. This method is essential for the network's operation, as it specifies how the input data flows through the network to produce the output.

### Agent Class

The Agent class utilizes a DQN to interact with an environmentm in a reinforcement learning context. It checks if CUDA (GPU support) is available and sets the appropriate device.

The Agent class is initialized with three parameters: 
- n_states, which represent the number of states
- n_actions, the number of actions
- hidden_size, the size of the hidden layers in the neural network
- learning_rate, the learning rate for the optimizer.

-------

The __init__ method first creates an instance of the QNetwork class, self.q_values, which is moved to the appropriate device (CPU or GPU) using the .to(device) method.

The class also initializes a mean squared error loss function (self.mse_loss) and an Adam optimizer (self.optimizer). The optimizer is configured to update the parameters of the local Q-network (self.q_values) with a predefined learning rate of 0.0001. Additionally, the class stores the number of states and actions as instance variables (self.n_states and self.n_actions).

---------

The __get_action__ method in the Agent class is responsible for selecting an action based on the current state and the epsilon-greedy policy. This method is crucial for balancing exploration and exploitation in reinforcement learning.

The method takes three parameters:

- state: A 2D tensor representing the current state, with a shape of (n, input_size).
- eps: A float representing the epsilon value used for epsilon-greedy action selection, which determines the probability of choosing a random action (exploration).
- check_eps: A boolean flag that, if set to False, bypasses the epsilon check and directly uses the policy network for action selection.

The method starts by generating a random sample using random.random(), which produces a float between 0.0 and 1.0. If check_eps is False or the random sample is greater than eps, the method proceeds with exploitation. It uses the policy network (self.model) to predict the Q-values for the given state. The state tensor is first converted to a Variable and then to the appropriate tensor type (FloatTensor). The method then selects the action with the highest Q-value using the index of the maximum value along dimension 1. This index is reshaped to a tensor and returned as the chosen action.

If the random sample is less than or equal to eps, the method opts for exploration and selects a random action. It chooses a random action index from the range of possible actions and returns it as a tensor on the appropriate device.

The condition ```if not check_eps or sample > eps```: checks whether the epsilon-greedy policy should be applied. If check_eps is False or the randomly generated sample is greater than the epsilon value (eps), the agent will exploit its current knowledge to choose the best action. This is done to balance exploration and exploitation, where a lower epsilon value encourages more exploitation. Within this condition, the with torch.no_grad(): context manager is used to disable gradient calculation. This is important because during action selection, we do not need to compute gradients, which saves memory and computational resources. It ensures that the operations within this block do not track gradients, making the forward pass more efficient. The self.model method, which represents the Q-network, is then called to compute the Q-values for the given state. The .data.max(1)[1] operation finds the index of the maximum Q-value along dimension 1, which corresponds to the best action. Finally, .view(1, 1) reshapes this index into a tensor of shape (1, 1) and returns it as the chosen action.

If the condition is not met, the else block is executed, indicating that the agent will explore. It generates a random action and returns it as a tensor. The random.randrange(self.n_actions) function selects a random integer from the range [0, self.n_actions), representing a random action index. This index is wrapped in a list of lists [[...]] to match the expected tensor shape and converted to a PyTorch tensor using torch.tensor(...). The device=device argument ensures that the tensor is created on the appropriate device (CPU or GPU).

Both return statements in the act method return the same type of data: a PyTorch tensor. Specifically, they return a tensor representing the chosen action.

Overall, the act method effectively implements the epsilon-greedy policy, allowing the agent to balance exploration and exploitation during its interaction with the environment.

---

The __learn__ method in the Agent class is responsible for updating the Q-network based on the agent's experience. This method implements the core of the Q-learning algorithm, which involves calculating the loss between the predicted Q-values and the target Q-values, and then updating the network's weights to minimize this loss.

The method takes six parameters:

- state: The current state of the environment.
- action: The action taken by the agent.
- reward: The reward received after taking the action.
- next_state: The state of the environment after the action is taken.
- done: A boolean flag indicating whether the episode has ended.
- gamma: The discount factor for future rewards.

First, the method computes the predicted Q-values for the given state and action using self.model(state).gather(1, action). The gather method is used to select the Q-values corresponding to the taken actions.

Next, the method calculates the maximum Q-values for the next state using self.model(next_state).max(1)[0].view(-1, 1). The max(1)[0] operation finds the maximum Q-value along dimension 1, which corresponds to the best action in the next state. The view(-1, 1) reshapes the tensor to ensure it has the correct dimensions.

The target Q-values are then computed using the Bellman equation: reward + (gamma * max_next_q_values * (1 - done)). This equation incorporates the immediate reward and the discounted maximum future reward, adjusted by the done flag to ensure that no future reward is considered if the episode has ended.

The loss between the predicted Q-values and the target Q-values is calculated using the mean squared error loss function: loss = self.mse_loss(q_values, target_q_values).

To update the network's weights, the method first resets the gradients of the model's parameters using self.optimizer.zero_grad(). This is necessary to prevent accumulation of gradients from previous updates. The loss.backward() call computes the gradients of the loss with respect to the model's parameters. Finally, self.optimizer.step() updates the model's parameters based on the computed gradients.

Overall, the learn method implements the Q-learning update rule, allowing the agent to improve its policy by minimizing the difference between predicted and target Q-values through gradient descent.

### Replay Buffer Class

### Jupyter Notebook
