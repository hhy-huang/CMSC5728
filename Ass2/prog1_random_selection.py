# This is the first program to simulate the multi-arm bandit
# Let say we only use RANDOM POLICY: each round, just randomly pick an arm
# Each arm has outcome 0 or 1, with probability 1 being the winning probability (Bernoulli distribution)

# Created by John C.S. Lui     Date: April 10, 2021


import numpy as np
from scipy.stats import bernoulli  # import bernoulli
import matplotlib.pyplot as plt

Num_of_Arms = 4      # number of arms

#  input parameters 
winning_parameters = np.array([0.2, 0.3, 0.85, 0.9], dtype=float)
max_prob = 0.9				      # record the highest probability of winning for all arms
optimal_arm = 3					      # index for the optimal arm
T = 1000					      # number of rounds to simulate
total_iteration = 200				      # number of iterations to the MAB simulation


reward_round_iteration = np.zeros((T), dtype=int)     # reward in each round average by # of iteration




# Go through T rounds, each round we need to select an arm


for iteration_count in range(total_iteration):
    for round in range(T):
        select_arm = np.random.randint(Num_of_Arms, size=1)   # randomly select an arm
    
        # generate reward for the selected arm
        reward = bernoulli.rvs(winning_parameters[select_arm]) 
        if reward == 1 :
            reward_round_iteration[round] += 1

        

# compute average reward for each  round

average_reward_in_each_round = np.zeros (T, dtype=float)

for round in range(T):
   average_reward_in_each_round[round] = float(reward_round_iteration[round])/float(total_iteration)
 
# Let generate X and Y data points to plot it out

cumulative_optimal_reward = 0.0
cumulative_reward = 0.0

X = np.zeros (T, dtype=int)
Y = np.zeros (T, dtype=float)
for round in range(T):
	X[round] = round
	cumulative_optimal_reward += max_prob
	cumulative_reward += average_reward_in_each_round[round]
	Y[round] = cumulative_optimal_reward - cumulative_reward

print('After ',T,'rounds, regret is: ', cumulative_optimal_reward - cumulative_reward)

#f = plt.figure()
#plt.plot(X, Y, color = 'red', ms = 5, label='linear regret')
#plt.ylim(ymin = 0)
#plt.xlabel('round number')
#plt.ylabel('regret')
#plt.title('Regret for Random Arm Selection policy')
#plt.legend()
#plt.grid(True)
#plt.xlim(0, T)
#plt.savefig("prog1_figure.png")
#plt.show()

fig, axs = plt.subplots(2)   # get two figures, top is regret, bottom is average reward in each round
fig.suptitle('Performance of Random Arm Selection')
fig.subplots_adjust(hspace=0.5)
axs[0].plot(X,Y, color = 'red', label='Regret of Random Arm Selection Policy')
axs[0].set(xlabel='round number', ylabel='Regret')
axs[0].grid(True)
axs[0].legend(loc='upper left')
axs[0].set_xlim(0,T)
axs[0].set_ylim(0,1.1*(cumulative_optimal_reward - cumulative_reward))
axs[1].plot(X, average_reward_in_each_round, color = 'black', label='average reward')
axs[1].set(xlabel='round number', ylabel='Average Reward per round')
axs[1].grid(True)
axs[1].legend(loc='upper left')
axs[1].set_xlim(0,T)
axs[1].set_ylim(0,1.0)
plt.savefig("prog1_figure.png")
plt.show()
