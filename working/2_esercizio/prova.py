import numpy as np
import matplotlib.pyplot as plt
import random

N=10 #number of doors
p=6 #number of doors opened by the host
trials = int(1e5)

def open_door(door, choice):
    # opens a door that is not the one chosen and does not contain a car
    open_doors = np.zeros(p)
    for i in range(p):
        open_doors[i] = random.choice([j for j in range(N) if j != choice and door[j] == 0])
    return open_doors

def switch_door(door, choice, opened):
    # switches to one of the other unopened doors
    return random.choice([i for i in range(N) if i != choice and i not in opened])

def newcomer_door(door, opened):
    # opens a door that is not opened
    return random.choice([i for i in range(N) if i not in opened])

conservative_wins = 0
switch_wins = 0
newcomer_wins = 0

# simulate the game for a number of trials

for _ in range(trials):

    # randomly place the car behind one of the doors
    door = np.zeros(N)
    door[random.randint(0, N-1)] = 1

    choice = random.randint(0, N-1)

    opened = open_door(door, choice)
    switcher_choice = switch_door(door, choice, opened)
    newcomer_choice = newcomer_door(door, opened)

    if door[choice] == 1:
        conservative_wins += 1
    if door[switcher_choice] == 1:
        switch_wins += 1
    if door[newcomer_choice] == 1:
        newcomer_wins += 1

print("total trials: ", trials)
print("Conservative wins: ", conservative_wins)
print("Switch wins: ", switch_wins)
print("Newcomer wins: ", newcomer_wins)

conservative_prob = conservative_wins / trials
switch_prob = switch_wins / trials
newcomer_prob = newcomer_wins / trials

probs = [conservative_prob, switch_prob, newcomer_prob]
labels = ['Conservative', 'Switcher', 'Newcomer']

plt.bar(labels, probs, color=['mediumblue', 'mediumpurple', 'darkmagenta'])
plt.ylabel('Probability of winning')
plt.title('Monty Hall Problem Simulation')
plt.ylim(0, 1)
plt.axhline(y=conservative_prob, color='mediumblue', linestyle='--', label=f'Conservative: {conservative_prob:.2f}')
plt.axhline(y=switch_prob, color='mediumpurple', linestyle='--', label=f'Switch: {switch_prob:.2f}')
plt.axhline(y=newcomer_prob, color='darkmagenta', linestyle='--', label=f'Newcomer: {newcomer_prob:.2f}')
plt.legend()
plt.show()