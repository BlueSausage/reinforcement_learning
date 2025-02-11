import random

rewards = [3, -8, 8, 10, -9, 7, 8, 9, 6, 1, -1, 7, 5, 8, 8, 2, -2, -5, 8, 7]
gamma = 0.0
total_return = 0

print(f"rewards: {rewards}")

for t in reversed(range(len(rewards))):
    print(f"reward: {rewards[t]}")
    total_return = rewards[t] + gamma * total_return
    print(f"total return: {total_return}")
    
# gamma 1: total return = total return: 72
# gamma 0.99: total return = 65.03259140787148
# gamma 0.5: total return = 2.1321163177490234
# gamma 0.1: total return = 2.2891789609758817
# gamma 0.0: total return = 3.0