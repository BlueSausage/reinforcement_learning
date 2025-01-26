## Comparison of the robustness of Deep Q-Networks (DQN) and REINFORCE in the event of sudden system changes in the LunarLander environment

### 1. Solving LunarLanding environment
Solving the LunarLanding environment, with Value- and Policy-Based learning, with turbulance (Gadgil et al., 2020; Guttulsrud et al., 2023).

### 2. Random Engine Failure
Another source of uncertainty in the physical world can be random engine failures due to the various unpredictable conditions in the agent’s environment. The model needs to be  robust enough to overcome such failures without impacting performance too much. To simulate this, we introduce action failure in the lunar lander. The agent takes the action provided 80% of the time, but 20% of the time the engines fail and it takes no action even though the provided action is firing an engine (Gadgil et al., 2020; Shah & Yao, 2023).

### LunarLanding

![Lunar Lander GIF](images/lunar_lander.gif "Lunar Lander")

#### Action Space
4 discrete action
- 0: do nothing
- 1: left engine
- 2: main engine
- 3: right engine

#### Observation Space
8-dimensional vector
- Position in x- and y-coordinates
- Velocity in x- und y-direction
- Angle
- Angle velocity
- One Boolean value per leg, for ground contact