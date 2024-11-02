"""
CMSC5728 Programming Assignment #2
Author: HUANG, Hao Yu
Date:   10/28/2024
"""
import random
import numpy as np
import math
import matplotlib.pyplot as plt


def cal_uni_expectation(upper, lower):
    """
    Target: calculate the expectation of the input uniform distribution
    Return: return the expectation
    """
    return (upper + lower) / 2

def arm_selection_random_selction_policy(num_of_arms):
    """
    Target: Conduct the random selection algorithm to simulate the multi-arm bandit
    Return: Return selected arm with the current algorithm
    """
    select_arm = np.random.randint(num_of_arms, size=1)   # randomly select an arm
    return select_arm


if __name__ == "__main__":
    print("=======================Random selection algorithm=======================")
    # parameters
    num_of_arms = 10                    # number of arms
    winning_parameters = np.array([tuple([0,2]), tuple([1,3]), tuple([2,4]), tuple([3,9]), tuple([4,6]), tuple([8,10]), tuple([3,5]), tuple([4,10]), tuple([5,7]), tuple([6,8])])
    # winning_parameters = np.array([0.2, 0.3, 0.85, 0.9], dtype=float)
    # max_prob = 0.9				        # record the highest probability of winning for all arms
    max_reward = np.max(np.array([cal_uni_expectation(x[1], x[0]) for x in winning_parameters]))
    optimal_arm = 5					    # index for the optimal arm
    T = 10000					        # number of rounds to simulate
    total_iteration = 200               # number of iterations to the MAB simulation

    # reward in each round average by # of iteration
    reward_round_iteration = np.zeros((T), dtype=int)

    # Go through T rounds, each round we need to select an arm
    for iteration_count in range(total_iteration):
        for round in range(T):
            # select the best arm with a specific algorithm
            select_arm = arm_selection_random_selction_policy(num_of_arms)[0]
            # generate reward for the selected arm
            # reward = bernoulli.rvs(winning_parameters[select_arm]) 
            reward = cal_uni_expectation(winning_parameters[select_arm][1], winning_parameters[select_arm][0])
            reward_round_iteration[round] += reward

    # compute average reward for each round
    average_reward_in_each_round = np.zeros (T, dtype=float)
    for round in range(T):
        average_reward_in_each_round[round] = float(reward_round_iteration[round])/float(total_iteration)
    
    # Let generate X and Y data points to plot it out
    cumulative_optimal_reward = 0.0
    cumulative_reward = 0.0
    X = np.zeros (T, dtype=int)
    Y = np.zeros (T, dtype=float)
    Y2 = np.zeros(T, dtype=float)
    for round in range(T):
        X[round] = round
        cumulative_optimal_reward += max_reward
        cumulative_reward += average_reward_in_each_round[round]
        Y2[round] = cumulative_reward
        Y[round] = (cumulative_optimal_reward - cumulative_reward)

    print(f'After {T} rounds:\n',\
          f'The average cumulative reward is: {cumulative_reward}\n',
          f'The average regret is: {(cumulative_optimal_reward - cumulative_reward)}'
          )
    
    fig, axs = plt.subplots(2)   # get two figures, top is regret, bottom is average reward in each round
    fig.suptitle('Performance of Random Arm Selection')
    fig.subplots_adjust(hspace=0.5)
    axs[0].plot(X,Y, color = 'red', label='Regret of Random Arm Selection Policy')
    axs[0].set(xlabel='round number', ylabel='Regret')
    axs[0].grid(True)
    axs[0].legend(loc='upper left')
    axs[0].set_xlim(0,T)
    axs[0].set_ylim(0,1.1*(cumulative_optimal_reward - cumulative_reward))
    axs[1].plot(X, Y2, color = 'black', label='cumulative average reward')
    axs[1].set(xlabel='round number', ylabel='Cumulative Average Reward per round')
    axs[1].grid(True)
    axs[1].legend(loc='upper left')
    axs[1].set_xlim(0,T)
    axs[1].set_ylim(0, max(Y2))
    plt.savefig("random.png")
    plt.show()