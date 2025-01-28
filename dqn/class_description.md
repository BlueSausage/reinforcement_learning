# RL Hausarbeit

Mögliche Themen
- Stabile Reglung
- entropy
- große schlagartige änderung im system
- im laufenden betrieb auf änderungen reagieren

Keywords:
- Uncertainty using Reinforcement Learning

Model predictive control

Verhalten nach dem schlag anschauen, closed loop regler, wie stark ist die 

wie lange kann eine engine ausbleiben (als eigene maße für die güte des reglers, maße beschreiben, daran rumschrauben, wie verhalten diese sich)

##### 1. Solving LunarLanding environment
Solving the LunarLanding environment, with Value- and Policy-Based learning, with turbulance (Gadgil et al., 2020; Guttulsrud et al., 2023).

##### 2. Random Engine Failure
Another source of uncertainty in the physical world can be random engine failures due to the various unpredictable conditions in the agent’s environment. The model needs to be  robust enough to overcome such failures without impacting performance too much. To simulate this, we introduce action failure in the lunar lander. The agent takes the action provided 80% of the time, but 20% of the time the engines fail and it takes no action even though the provided action is firing an engine (Gadgil et al., 2020; Shah & Yao, 2023).

## Vergleich der Robustheit von Deep Q-Networks (DQN) und REINFORCE bei plötzlichen Systemänderungen in der LunarLander-Umgebung

### Gliederung
1. Einleitung
- Hintergrund:
    - Einführung in Reinforcement Learning (RL) und die zentrale Herausforderung der Unsicherheiten in der realen Welt.
    - Relevanz von Robustheit in RL-Agenten in physikalischen Systemen.
    - Zielsetzung: Untersuchung der Anpassungsfähigkeit und Robustheit von wertbasierten (DQN) und policybasierten (REINFORCE) RL-Ansätzen bei plötzlichen Triebwerksausfällen.
- Motivation:
    - Wie beeinflussen Systemänderungen wie Triebwerksausfälle die Leistung?
    - Warum ist es wichtig, robuste Algorithmen zu entwickeln?
- Fragestellungen/Hypothesen:
    - Wie unterscheiden sich die beiden Ansätze in ihrer Fähigkeit, plötzliche Systemänderungen zu bewältigen?
    - Zeigt ein Ansatz eine stabilere Leistung oder schnellere Anpassung?

2. Grundlagen
    1. Reinforcement Learning Grundlagen:
    - Wertbasierte Methode: Deep Q-Network (DQN).
    - Policy-Gradient-Methode: REINFORCE.
    - Ziel: Erwartungswert maximieren und optimale Politik finden.
    2. Robustheit in Reinforcement Learning:
    - Bedeutung von Robustheit und Anpassungsfähigkeit.
    - Definition von Unsicherheiten in RL (z. B. stochastische Dynamiken, externe Störungen).
    3. LunarLander-Umgebung:
    - Beschreibung der Umgebung und ihrer Dynamiken.
    - Technische Details (OpenAI Gym, State-Space, Actions).
    - Simulation von Triebwerksausfällen (z. B. Deaktivierung eines Triebwerks während des Trainings oder Tests)

3. Methodik
    1. Implementierung der RL-Ansätze:
    - Wertbasierte Methode: DQN.
    - Policy-Gradient-Methode: REINFORCE.
    - Tools/Bibliotheken: OpenAI Gym, TensorFlow oder PyTorch, NumPy, Matplotlib.
    2. Simulation von Unsicherheiten:
    - Einführung eines Triebwerksausfalls: Deaktivierung eines der vier Triebwerke während eines zufälligen Zeitschritts.
    - Vergleich von Training ohne Ausfälle und Training mit zufälligen Ausfällen.
    3. Metriken zur Bewertung:
    - Stabilität: Durchschnittliche Belohnung nach der Störung.
    - Robustheit: Wie schnell passt sich der Algorithmus an (gemessen an der Lernrate nach der Änderung)?
    - Performance-Verlust: Unterschied in der durchschnittlichen Belohnung vor und nach der Störung.
    - Konvergenzgeschwindigkeit: Zeit, die benötigt wird, um eine stabile Politik nach einer Störung zu entwickeln.

4. Experimente und Ergebnisse
    1. Training ohne Triebwerksausfälle:
    - Vergleich der Leistung von DQN und REINFORCE in der Standardumgebung.
    2. Training mit zufälligen Triebwerksausfällen:
    - Beschreibung des Experiments (wie, wann und welche Ausfälle simuliert werden).
    - Auswertung der Lernkurven: Wie wirkt sich der Ausfall auf die Leistung aus?
    3. Vergleich der Ansätze:
    - Stabilität und Robustheit unter Systemänderungen.
    - Analyse, ob ein Ansatz die Änderungen besser bewältigt als der andere.
    4. Zusätzliche Beobachtungen:
    - Verhalten bei wiederholten oder langfristigen Störungen.
    - Unterschiede in der Exploration/Exploitation-Strategie der beiden Ansätze.

5. Diskussion
    1. Interpretation der Ergebnisse:
    - Warum zeigt ein Ansatz möglicherweise bessere Anpassungsfähigkeit?
    - Rolle der Modellarchitektur (z. B. neuronales Netzwerk in DQN vs. stochastische Politik in Policy-Gradient-Methoden).
    2. Relevanz der Ergebnisse:
    - Bedeutung der Robustheit in physischen Systemen.
    - Übertragbarkeit auf andere reale Szenarien (z. B. Robotik, autonome Fahrzeuge).
    3. Limitierungen der Experimente:
    - Einschränkungen der LunarLander-Umgebung als Modell.
    - Potentielle Herausforderungen bei komplexeren oder realistischeren Umgebungen.
    4. Verbesserungsvorschläge:
    - Erweiterte Algorithmen wie PPO, TRPO, oder Double DQN.
    - Hinzufügen von Mechanismen zur Fehlerkompensation (z. B. Backup-Strategien für den Agenten).

6. Fazit und Ausblick
    - Zusammenfassung der Ergebnisse:
        - Wie gut reagieren die Algorithmen auf plötzliche Änderungen im System?
        - Welcher Ansatz zeigt robustere und stabilere Leistungen?
    - Praktische Implikationen:
        - Bedeutung der Ergebnisse für reale Anwendungen.
    - Zukünftige Arbeiten:
        - Test auf komplexere Umgebungen oder andere Arten von Störungen (z. B. Windböen, veränderte Gravitationsbedingungen).
        - Einsatz fortgeschrittener Algorithmen.


### LunarLanding

![Lunar Lander GIF](images/lunar_lander.gif "Lunar Lander")

#### Action Space
4 diskrete Aktionen
- 0: nichts machen
- 1: linker Motor
- 2: haupt Motor
- 3: rechter Motor

#### Observation Space
8-dimensionaler Vektor
- Position in x- und y-Koordinaten
- Geschwindigkeit in x- und y-Richtung
- Winkel
- Winkelgeschwindigkeit
- jeweils ein Boolean-Wert pro Bein, für Bodenkontakt

## Code description

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

### Jupyter Notebook
