### Agent Class

The Agent class utilizes a policy network to interact with an environment in a reinforcement learning context. This agent follows the REINFORCE algorithm, a type of policy gradient method.

The Agent class is initialized with five parameters:

- n_states: The number of states in the environment.
- n_actions: The number of possible actions.
- hidden_size: The size of the hidden layers in the neural network.
- learning_rate: The learning rate for the optimizer.
- gamma: The discount factor for future rewards.


#### init-Method

The __init__ method first creates an instance of the PolicyNetwork class, self.policy, which is moved to the appropriate device (CPU or GPU). The class also initializes an Adam optimizer, which is configured to update the parameters of the policy network with the specified learning rate. Additionally, the discount factor gamma is stored as an instance variable.

The get_action method is responsible for selecting an action based on the current state using the policy network. It takes one parameter:

- state: A 2D tensor representing the current state.

#### get_action-Method

The __get_action__ method in the Agent class is responsible for selecting an action based on the current state using the policy network. This method is crucial for determining the agent's behavior in the environment according to the learned policy.

The method takes one parameter:

- state: A 2D tensor representing the current state of the environment, with a shape of (n, input_size).

The method begins by using the with torch.no_grad(): context manager to disable gradient calculation. This is important because during action selection, we do not need to compute gradients, which saves memory and computational resources. It ensures that the operations within this block do not track gradients, making the forward pass more efficient.

Within this context, the method computes the action probabilities using the policy network: probs = self.policy(state.float()).multinomial(1). The state tensor is first converted to a float tensor to match the expected input type for the policy network. The multinomial(1) function is called on the output of the policy network. The multinomial function samples from the probability distribution provided by the policy network. The argument 1 specifies that one sample (one action) should be drawn from the distribution. The result is a tensor containing the sampled action based on the computed probabilities.

Next, a Categorical distribution is created using these probabilities. The Categorical class represents a categorical distribution parameterized by either probabilities (probs) or log probabilities (logits). In this case, it is parameterized by the action probabilities computed by the policy network.

The method then samples an action from this distribution: action = dist.sample(). The sample() method generates a random sample (an action) from the categorical distribution based on the provided probabilities.

Finally, the method returns the chosen action as an integer (action.item()) and the log probability of the action (dist.log_prob(action)). The item() method extracts the value of the action tensor as a standard Python number, while the log_prob() method computes the log probability of the sampled action. The log probability is useful for calculating the policy gradient during the learning phase.

Overall, the get_action method effectively selects an action based on the current policy by leveraging the policy network's predictions and the Categorical distribution. This approach allows the agent to sample actions according to the learned policy, facilitating exploration and exploitation in the environment.

#### learn-Method

The learn method in the Agent class is responsible for updating the policy network using a batch of rewards and log probabilities. This method implements the policy gradient update rule, which is a core component of the REINFORCE algorithm in reinforcement learning.

The method takes two parameters:

- rewards: A list of rewards received by the agent.
- log_probs: A list of log probabilities of the actions taken by the agent.

The method first calculates the returns, which are the cumulative discounted rewards. It initializes an empty list returns and a variable G to store the cumulative return. It then iterates over the rewards in reverse order (using rewards[::-1]) to compute the returns. For each reward r, it updates G using the formula G = r + self.gamma * G and inserts G at the beginning of the returns list.

The method then converts the returns list to a PyTorch tensor and moves it to the appropriate device. It then normalizes the returns by subtracting the mean and dividing by the standard deviation plus a small constant (1e-8) to avoid division by zero. Normalizing the returns helps stabilize training by ensuring that the scale of the returns is consistent.

Then the method initializes an empty list policy_loss to store the loss values. It then iterates over the log probabilities and the corresponding returns using the zip function. For each pair of log_prob and G, it computes the policy loss as -log_prob * G and appends it to the policy_loss list. The negative sign is used because the goal is to maximize the expected return, and the optimizer minimizes the loss.

The method performs the following steps to update the policy network:

1. It calls self.optimizer.zero_grad() to reset the gradients of the model's parameters. This is necessary to prevent accumulation of gradients from previous updates.
2. It concatenates the policy_loss list into a single tensor using torch.cat(policy_loss).sum(). The sum() function aggregates the loss values into a single scalar.
3. It calls policy_loss.backward() to compute the gradients of the loss with respect to the model's parameters.
4. It calls self.optimizer.step() to update the model's parameters based on the computed gradients.

Overall, the learn method updates the policy network by computing the policy gradient and performing a gradient descent step. This process allows the agent to improve its policy based on the observed rewards and the actions taken, facilitating learning and adaptation in the environment.