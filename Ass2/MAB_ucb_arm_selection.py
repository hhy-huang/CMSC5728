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

def arm_selection_ucb_policy(num_of_arms, pull_counts, rewards, round_num):
    """
    Target: Conduct the UCB selection algorithm to simulate the multi-arm bandit
    Return: Return selected arm with the current algorithm
    """
    # Calculate the average reward and the number of pulls for each arm
    averages = rewards / pull_counts
    counts = pull_counts
    
    # Calculate the UCB index for each arm
    ucb_values = averages + math.sqrt(2 * math.log(round_num) / counts)
    
    # Select the arm with the highest UCB index
    select_arm = np.argmax(ucb_values)
    
    return select_arm

if __name__ == "__main__":
    print("=======================UCB algorithm=======================")
    # parameters
    num_of_arms = 10                    # number of arms
    winning_parameters = np.array([tuple([0,2]), tuple([1,3]), tuple([2,4]), tuple([3,9]), tuple([4,6]), tuple([8,10]), tuple([3,5]), tuple([4,10]), tuple([5,7]), tuple([6,8])])
    T = 10000					        # number of rounds to simulate
    total_iteration = 200               # number of iterations to the MAB simulation

    # Initialize arrays to store the number of pulls and rewards for each arm
    pull_counts = np.zeros(num_of_arms, dtype=int)
    rewards = np.zeros(num_of_arms, dtype=float)

    # reward in each round average by # of iteration
    reward_round_iteration = np.zeros((T), dtype=float)

    # Go through T rounds, each round we need to select an arm
    for iteration_count in range(total_iteration):
        for round in range(T):
            # select the best arm with a specific algorithm
            select_arm = arm_selection_ucb_policy(num_of_arms, pull_counts, rewards, round + T * iteration_count)[0]
            # generate reward for the selected arm
            reward = cal_uni_expectation(winning_parameters[select_arm][1], winning_parameters[select_arm][0])
            rewards[select_arm] += reward
            reward_round_iteration[round] += reward
            pull_counts[select_arm] += 1

    # compute average reward for each round
    average_reward_in_each_round = np.zeros(T, dtype=float)
    for round in range(T):
        average_reward_in_each_round[round] = reward_round_iteration[round] / total_iteration
    
    # Let generate X and Y data points to plot it out
    cumulative_optimal_reward = 0.0
    cumulative_reward = 0.0
    X = np.zeros(T, dtype=int)
    Y = np.zeros(T, dtype=float)
    for round in range(T):
        X[round] = round
        cumulative_optimal_reward += cal_uni_expectation(winning_parameters[5][1], winning_parameters[5][0])
        cumulative_reward += average_reward_in_each_round[round]
        Y[round] = cumulative_optimal_reward - cumulative_reward

    print(f'After {T} rounds:\n',
          f'The average cumulative reward is: {cumulative_reward}\n',
          f'The average regret is: {(cumulative_optimal_reward - cumulative_reward)}'
          )
    
    fig, axs = plt.subplots(2)   # get two figures, top is regret, bottom is average reward in each round
    fig.suptitle('Performance of UCB Arm Selection')
    fig.subplots_adjust(hspace=0.5)
    axs[0].plot(X, Y, color='red', label='Regret of UCB Arm Selection Policy')
    axs[0].set(xlabel='round number', ylabel='Regret')
    axs[0].grid(True)
    axs[0].legend(loc='upper left')
    axs[0].set_xlim(0, T)
    axs[0].set_ylim(0, 1.1 * (cumulative_optimal_reward - cumulative_reward))
    axs[1].plot(X, average_reward_in_each_round, color='black', label='average reward')
    axs[1].set(xlabel='round number', ylabel='Average Reward per round')
    axs[1].grid(True)
    axs[1].legend(loc='upper left')
    axs[1].set_xlim(0, T)
    axs[1].set_ylim(0, 10.0)
    plt.savefig("ucb.png")
    plt.show()