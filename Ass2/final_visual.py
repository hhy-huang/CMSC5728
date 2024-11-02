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

def epsilon_fuc(t):
    # Define epsilon as a function of time t
    return 1.0 / (1.0 + t)

def arm_selection_random_selction_policy(num_of_arms):
    """
    Target: Conduct the random selection algorithm to simulate the multi-arm bandit
    Return: Return selected arm with the current algorithm
    """
    select_arm = np.random.randint(num_of_arms, size=1)   # randomly select an arm
    return select_arm

def arm_selection_ucb_policy(num_of_arms, pull_counts, rewards, t):
    """
    Target: Conduct the UCB selection algorithm to simulate the multi-arm bandit
    Return: Return selected arm with the current algorithm
    """
    # Calculate the average reward for each arm
    averages = rewards / pull_counts
    
    # Calculate the UCB index for each arm
    ucb_values = np.array([averages + math.sqrt(2 * math.log(t) / x) for x in pull_counts])
    
    # Select the arm with the highest UCB index
    select_arm = np.argmax(ucb_values)
    
    return select_arm

def arm_selection_epsilon_greedy_policy(num_of_arms, epsilon, pull_counts, rewards):
    """
    Target: Conduct the epsilon-greedy selection algorithm to simulate the multi-arm bandit
    Return: Return selected arm with the current algorithm
    """
    if random.random() < epsilon:
        select_arm = np.random.randint(num_of_arms, size=1)[0]  # Explore: randomly select an arm
    else:
        select_arm = np.argmax(rewards / (pull_counts + 1e-10))  # Exploit: select the arm with the highest average reward
    pull_counts[select_arm] += 1
    return select_arm

def arm_selection_ete_policy(round, num_of_arms, exploration_phase, current_best_arm, pull_counts, rewards):
    """
    Target: Conduct the Explore-Then-Exploit selection algorithm to simulate the multi-arm bandit
    Return: Return selected arm with the current algorithm
    """
    if exploration_phase:
        # In the exploration phase, try each arm once
        # select_arm = np.random.randint(num_of_arms, size=1)[0]
        select_arm = round % (num_of_arms)
        pull_counts[select_arm] += 1
    else:
        # In the exploitation phase, always choose the best arm
        select_arm = current_best_arm
        pull_counts[select_arm] += 1
    return select_arm


if __name__ == "__main__":
    T = 10000					        # number of rounds to simulate
    num_of_arms = 10
    winning_parameters = np.array([tuple([0,2]), tuple([1,9]), tuple([1,3]), tuple([3,5]), tuple([4,6]), tuple([2,10]), tuple([1,5]), tuple([4,10]), tuple([5,7]), tuple([0,10])])
    total_iteration = 200               # number of iterations to the MAB simulation
    optimal_arm = 7
    reward_avg = np.sum(np.array([cal_uni_expectation(x[1], x[0]) for x in winning_parameters])) / num_of_arms
    # random
    print("=======================Random selection algorithm=======================")
    # parameters
    max_reward = np.max(np.array([cal_uni_expectation(x[1], x[0]) for x in winning_parameters]))

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
    Y_random = np.zeros (T, dtype=float)
    Y2_random = np.zeros(T, dtype=float)
    for round in range(T):
        X[round] = round
        cumulative_optimal_reward += max_reward
        cumulative_reward += average_reward_in_each_round[round]
        Y2_random[round] = cumulative_reward
        Y_random[round] = (cumulative_optimal_reward - cumulative_reward)

    print(f'After {T} rounds:\n',\
          f'The average cumulative reward is: {cumulative_reward}\n',
          f'The average regret is: {(cumulative_optimal_reward - cumulative_reward)}'
          )

    # ucb
    print("=======================UCB algorithm=======================")
    # Initialize arrays to store the number of pulls and rewards for each arm
    pull_counts = np.zeros(num_of_arms, dtype=int)
    rewards = np.zeros(num_of_arms, dtype=float)

    # reward in each round average by # of iteration
    reward_round_iteration = np.zeros((T), dtype=float)

    # Go through T rounds, each round we need to select an arm
    for iteration_count in range(total_iteration):
        for round in range(T):
            t = round + T * iteration_count + 1  # current time slot
            if round < num_of_arms:
                # First N rounds, pull each arm once
                select_arm = round
            else:
                # After the first N rounds, use UCB policy to select the arm
                select_arm = arm_selection_ucb_policy(num_of_arms, pull_counts, rewards, t)
            
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
    Y_ucb = np.zeros(T, dtype=float)
    Y2_ucb = np.zeros(T, dtype=float)
    for round in range(T):
        cumulative_optimal_reward += cal_uni_expectation(winning_parameters[optimal_arm][1], winning_parameters[optimal_arm][0])
        cumulative_reward += average_reward_in_each_round[round]
        Y2_ucb[round] = cumulative_reward
        Y_ucb[round] = cumulative_optimal_reward - cumulative_reward

    print(f'After {T} rounds:\n',
          f'The average cumulative reward is: {cumulative_reward}\n',
          f'The average regret is: {(cumulative_optimal_reward - cumulative_reward)}'
          )
    
    # greedy arm
    print("=======================Epsilon-Greedy algorithm=======================")
    # parameters
    epsilon = 0.1                       # exploration rate

    # Initialize arrays to store the number of pulls and rewards for each arm
    pull_counts = np.zeros(num_of_arms, dtype=int)
    rewards = np.zeros(num_of_arms, dtype=float)

    # reward in each round average by # of iteration
    reward_round_iteration = np.zeros((T), dtype=float)

    # Go through T rounds, each round we need to select an arm
    for iteration_count in range(total_iteration):
        for round in range(T):
            # select the best arm with a specific algorithm
            select_arm = arm_selection_epsilon_greedy_policy(num_of_arms, epsilon, pull_counts, rewards)
            # generate reward for the selected arm
            reward = cal_uni_expectation(winning_parameters[select_arm][1], winning_parameters[select_arm][0])
            rewards[select_arm] += reward
            reward_round_iteration[round] += reward

    # compute average reward for each round
    average_reward_in_each_round = np.zeros(T, dtype=float)
    for round in range(T):
        average_reward_in_each_round[round] = reward_round_iteration[round] / total_iteration
    
    # Let generate X and Y data points to plot it out
    cumulative_optimal_reward = 0.0
    cumulative_reward = 0.0
    Y_greedy_01 = np.zeros(T, dtype=float)
    Y2_greedy_01 = np.zeros(T, dtype=float)
    for round in range(T):
        cumulative_optimal_reward += cal_uni_expectation(winning_parameters[optimal_arm][1], winning_parameters[optimal_arm][0])
        cumulative_reward += average_reward_in_each_round[round]
        Y2_greedy_01[round] = cumulative_reward
        Y_greedy_01[round] = cumulative_optimal_reward - cumulative_reward

    print(f'After {T} rounds:\n',
          f'The average cumulative reward is: {cumulative_reward}\n',
          f'The average regret is: {(cumulative_optimal_reward - cumulative_reward)}'
          )
    
    print("=======================Epsilon-Greedy algorithm=======================")
    # parameters
    epsilon = 0.2                       # exploration rate

    # Initialize arrays to store the number of pulls and rewards for each arm
    pull_counts = np.zeros(num_of_arms, dtype=int)
    rewards = np.zeros(num_of_arms, dtype=float)

    # reward in each round average by # of iteration
    reward_round_iteration = np.zeros((T), dtype=float)

    # Go through T rounds, each round we need to select an arm
    for iteration_count in range(total_iteration):
        for round in range(T):
            # select the best arm with a specific algorithm
            select_arm = arm_selection_epsilon_greedy_policy(num_of_arms, epsilon, pull_counts, rewards)
            # generate reward for the selected arm
            reward = cal_uni_expectation(winning_parameters[select_arm][1], winning_parameters[select_arm][0])
            rewards[select_arm] += reward
            reward_round_iteration[round] += reward

    # compute average reward for each round
    average_reward_in_each_round = np.zeros(T, dtype=float)
    for round in range(T):
        average_reward_in_each_round[round] = reward_round_iteration[round] / total_iteration
    
    # Let generate X and Y data points to plot it out
    cumulative_optimal_reward = 0.0
    cumulative_reward = 0.0
    Y_greedy_02 = np.zeros(T, dtype=float)
    Y2_greedy_02 = np.zeros(T, dtype=float)
    for round in range(T):
        cumulative_optimal_reward += cal_uni_expectation(winning_parameters[optimal_arm][1], winning_parameters[optimal_arm][0])
        cumulative_reward += average_reward_in_each_round[round]
        Y2_greedy_02[round] = cumulative_reward
        Y_greedy_02[round] = cumulative_optimal_reward - cumulative_reward

    print(f'After {T} rounds:\n',
          f'The average cumulative reward is: {cumulative_reward}\n',
          f'The average regret is: {(cumulative_optimal_reward - cumulative_reward)}'
          )
    
    print("=======================Epsilon-Greedy algorithm=======================")
    # parameters
    epsilon = 0.3                       # exploration rate

    # Initialize arrays to store the number of pulls and rewards for each arm
    pull_counts = np.zeros(num_of_arms, dtype=int)
    rewards = np.zeros(num_of_arms, dtype=float)

    # reward in each round average by # of iteration
    reward_round_iteration = np.zeros((T), dtype=float)

    # Go through T rounds, each round we need to select an arm
    for iteration_count in range(total_iteration):
        for round in range(T):
            # select the best arm with a specific algorithm
            select_arm = arm_selection_epsilon_greedy_policy(num_of_arms, epsilon, pull_counts, rewards)
            # generate reward for the selected arm
            reward = cal_uni_expectation(winning_parameters[select_arm][1], winning_parameters[select_arm][0])
            rewards[select_arm] += reward
            reward_round_iteration[round] += reward

    # compute average reward for each round
    average_reward_in_each_round = np.zeros(T, dtype=float)
    for round in range(T):
        average_reward_in_each_round[round] = reward_round_iteration[round] / total_iteration
    
    # Let generate X and Y data points to plot it out
    cumulative_optimal_reward = 0.0
    cumulative_reward = 0.0
    Y_greedy_03 = np.zeros(T, dtype=float)
    Y2_greedy_03 = np.zeros(T, dtype=float)
    for round in range(T):
        cumulative_optimal_reward += cal_uni_expectation(winning_parameters[optimal_arm][1], winning_parameters[optimal_arm][0])
        cumulative_reward += average_reward_in_each_round[round]
        Y2_greedy_03[round] = cumulative_reward
        Y_greedy_03[round] = cumulative_optimal_reward - cumulative_reward

    print(f'After {T} rounds:\n',
          f'The average cumulative reward is: {cumulative_reward}\n',
          f'The average regret is: {(cumulative_optimal_reward - cumulative_reward)}'
          )
    
    print("=======================Epsilon-Greedy algorithm=======================")
    # parameters
    epsilon = 0.4                       # exploration rate

    # Initialize arrays to store the number of pulls and rewards for each arm
    pull_counts = np.zeros(num_of_arms, dtype=int)
    rewards = np.zeros(num_of_arms, dtype=float)

    # reward in each round average by # of iteration
    reward_round_iteration = np.zeros((T), dtype=float)

    # Go through T rounds, each round we need to select an arm
    for iteration_count in range(total_iteration):
        for round in range(T):
            # select the best arm with a specific algorithm
            select_arm = arm_selection_epsilon_greedy_policy(num_of_arms, epsilon, pull_counts, rewards)
            # generate reward for the selected arm
            reward = cal_uni_expectation(winning_parameters[select_arm][1], winning_parameters[select_arm][0])
            rewards[select_arm] += reward
            reward_round_iteration[round] += reward

    # compute average reward for each round
    average_reward_in_each_round = np.zeros(T, dtype=float)
    for round in range(T):
        average_reward_in_each_round[round] = reward_round_iteration[round] / total_iteration
    
    # Let generate X and Y data points to plot it out
    cumulative_optimal_reward = 0.0
    cumulative_reward = 0.0
    Y_greedy_04 = np.zeros(T, dtype=float)
    Y2_greedy_04 = np.zeros(T, dtype=float)
    for round in range(T):
        cumulative_optimal_reward += cal_uni_expectation(winning_parameters[optimal_arm][1], winning_parameters[optimal_arm][0])
        cumulative_reward += average_reward_in_each_round[round]
        Y2_greedy_04[round] = cumulative_reward
        Y_greedy_04[round] = cumulative_optimal_reward - cumulative_reward

    print(f'After {T} rounds:\n',
          f'The average cumulative reward is: {cumulative_reward}\n',
          f'The average regret is: {(cumulative_optimal_reward - cumulative_reward)}'
          )
    
    print("=======================Epsilon-Greedy algorithm=======================")
    # parameters
    epsilon = 0.5                       # exploration rate

    # Initialize arrays to store the number of pulls and rewards for each arm
    pull_counts = np.zeros(num_of_arms, dtype=int)
    rewards = np.zeros(num_of_arms, dtype=float)

    # reward in each round average by # of iteration
    reward_round_iteration = np.zeros((T), dtype=float)

    # Go through T rounds, each round we need to select an arm
    for iteration_count in range(total_iteration):
        for round in range(T):
            # select the best arm with a specific algorithm
            select_arm = arm_selection_epsilon_greedy_policy(num_of_arms, epsilon, pull_counts, rewards)
            # generate reward for the selected arm
            reward = cal_uni_expectation(winning_parameters[select_arm][1], winning_parameters[select_arm][0])
            rewards[select_arm] += reward
            reward_round_iteration[round] += reward

    # compute average reward for each round
    average_reward_in_each_round = np.zeros(T, dtype=float)
    for round in range(T):
        average_reward_in_each_round[round] = reward_round_iteration[round] / total_iteration
    
    # Let generate X and Y data points to plot it out
    cumulative_optimal_reward = 0.0
    cumulative_reward = 0.0
    Y_greedy_05 = np.zeros(T, dtype=float)
    Y2_greedy_05 = np.zeros(T, dtype=float)
    for round in range(T):
        cumulative_optimal_reward += cal_uni_expectation(winning_parameters[optimal_arm][1], winning_parameters[optimal_arm][0])
        cumulative_reward += average_reward_in_each_round[round]
        Y2_greedy_05[round] = cumulative_reward
        Y_greedy_05[round] = cumulative_optimal_reward - cumulative_reward

    print(f'After {T} rounds:\n',
          f'The average cumulative reward is: {cumulative_reward}\n',
          f'The average regret is: {(cumulative_optimal_reward - cumulative_reward)}'
          )
    
    # ETE
    print("=======================Explore-Then-Exploit algorithm=======================")
    # parameters
    m = 1
    exploration_phase_length = m * num_of_arms      # number of rounds in the exploration phase

    # Initialize arrays to store the number of pulls and rewards for each arm
    pull_counts = np.zeros(num_of_arms, dtype=int)
    rewards = np.zeros(num_of_arms, dtype=float)

    # reward in each round average by # of iteration
    reward_round_iteration = np.zeros((T), dtype=int)

    # Initialize the best arm as the first arm
    current_best_arm = 0
    current_best_reward = cal_uni_expectation(winning_parameters[current_best_arm][1], winning_parameters[current_best_arm][0])

    for iteration_count in range(total_iteration):
        exploration_phase = True
        for round in range(T):
            # select the best arm with a specific algorithm
            select_arm = arm_selection_ete_policy(round, num_of_arms, exploration_phase, current_best_arm, pull_counts, rewards)
            # generate reward for the selected arm
            reward = cal_uni_expectation(winning_parameters[select_arm][1], winning_parameters[select_arm][0])
            reward_round_iteration[round] += reward
            rewards[select_arm] += reward
            if pull_counts[select_arm] == 1:
                # Update the best arm after each arm has been tried once
                if reward > current_best_reward:
                    current_best_arm = select_arm
                    current_best_reward = reward
            if round >= exploration_phase_length - 1:
                exploration_phase = False

    # compute average reward for each round
    average_reward_in_each_round = np.zeros(T, dtype=float)
    for round in range(T):
        average_reward_in_each_round[round] = float(reward_round_iteration[round])/float(total_iteration)
    
    # Let generate X and Y data points to plot it out
    cumulative_optimal_reward = 0.0
    cumulative_reward = 0.0
    Y_ete_1 = np.zeros(T, dtype=float)
    Y2_ete_1 = np.zeros(T, dtype=float)
    for round in range(T):
        cumulative_optimal_reward += cal_uni_expectation(winning_parameters[optimal_arm][1], winning_parameters[optimal_arm][0])
        cumulative_reward += average_reward_in_each_round[round]
        Y2_ete_1[round] = cumulative_reward
        Y_ete_1[round] = (cumulative_optimal_reward - cumulative_reward)

    print(f'After {T} rounds:\n',
          f'The average cumulative reward is: {cumulative_reward}\n',
          f'The average regret is: {(cumulative_optimal_reward - cumulative_reward)}'
          )

    print("=======================Explore-Then-Exploit algorithm=======================")
    m = 10
    exploration_phase_length = m * num_of_arms      # number of rounds in the exploration phase

    # Initialize arrays to store the number of pulls and rewards for each arm
    pull_counts = np.zeros(num_of_arms, dtype=int)
    rewards = np.zeros(num_of_arms, dtype=float)

    # reward in each round average by # of iteration
    reward_round_iteration = np.zeros((T), dtype=int)

    # Initialize the best arm as the first arm
    current_best_arm = 0
    current_best_reward = cal_uni_expectation(winning_parameters[current_best_arm][1], winning_parameters[current_best_arm][0])

    for iteration_count in range(total_iteration):
        exploration_phase = True
        for round in range(T):
            # select the best arm with a specific algorithm
            select_arm = arm_selection_ete_policy(round, num_of_arms, exploration_phase, current_best_arm, pull_counts, rewards)
            # generate reward for the selected arm
            reward = cal_uni_expectation(winning_parameters[select_arm][1], winning_parameters[select_arm][0])
            reward_round_iteration[round] += reward
            rewards[select_arm] += reward
            if pull_counts[select_arm] == 1:
                # Update the best arm after each arm has been tried once
                if reward > current_best_reward:
                    current_best_arm = select_arm
                    current_best_reward = reward
            if round >= exploration_phase_length - 1:
                exploration_phase = False

    # compute average reward for each round
    average_reward_in_each_round = np.zeros(T, dtype=float)
    for round in range(T):
        average_reward_in_each_round[round] = float(reward_round_iteration[round])/float(total_iteration)
    
    # Let generate X and Y data points to plot it out
    cumulative_optimal_reward = 0.0
    cumulative_reward = 0.0
    Y_ete_10 = np.zeros(T, dtype=float)
    Y2_ete_10 = np.zeros(T, dtype=float)
    for round in range(T):
        cumulative_optimal_reward += cal_uni_expectation(winning_parameters[optimal_arm][1], winning_parameters[optimal_arm][0])
        cumulative_reward += average_reward_in_each_round[round]
        Y2_ete_10[round] = cumulative_reward
        Y_ete_10[round] = (cumulative_optimal_reward - cumulative_reward)

    print(f'After {T} rounds:\n',
          f'The average cumulative reward is: {cumulative_reward}\n',
          f'The average regret is: {(cumulative_optimal_reward - cumulative_reward)}'
          )
    
    print("=======================Explore-Then-Exploit algorithm=======================")
    m = 20
    exploration_phase_length = m * num_of_arms      # number of rounds in the exploration phase

    # Initialize arrays to store the number of pulls and rewards for each arm
    pull_counts = np.zeros(num_of_arms, dtype=int)
    rewards = np.zeros(num_of_arms, dtype=float)

    # reward in each round average by # of iteration
    reward_round_iteration = np.zeros((T), dtype=int)

    # Initialize the best arm as the first arm
    current_best_arm = 0
    current_best_reward = cal_uni_expectation(winning_parameters[current_best_arm][1], winning_parameters[current_best_arm][0])

    for iteration_count in range(total_iteration):
        exploration_phase = True
        for round in range(T):
            # select the best arm with a specific algorithm
            select_arm = arm_selection_ete_policy(round, num_of_arms, exploration_phase, current_best_arm, pull_counts, rewards)
            # generate reward for the selected arm
            reward = cal_uni_expectation(winning_parameters[select_arm][1], winning_parameters[select_arm][0])
            reward_round_iteration[round] += reward
            rewards[select_arm] += reward
            if pull_counts[select_arm] == 1:
                # Update the best arm after each arm has been tried once
                if reward > current_best_reward:
                    current_best_arm = select_arm
                    current_best_reward = reward
            if round >= exploration_phase_length - 1:
                exploration_phase = False

    # compute average reward for each round
    average_reward_in_each_round = np.zeros(T, dtype=float)
    for round in range(T):
        average_reward_in_each_round[round] = float(reward_round_iteration[round])/float(total_iteration)
    
    # Let generate X and Y data points to plot it out
    cumulative_optimal_reward = 0.0
    cumulative_reward = 0.0
    Y_ete_20 = np.zeros(T, dtype=float)
    Y2_ete_20 = np.zeros(T, dtype=float)
    for round in range(T):
        cumulative_optimal_reward += cal_uni_expectation(winning_parameters[optimal_arm][1], winning_parameters[optimal_arm][0])
        cumulative_reward += average_reward_in_each_round[round]
        Y2_ete_20[round] = cumulative_reward
        Y_ete_20[round] = (cumulative_optimal_reward - cumulative_reward)

    print(f'After {T} rounds:\n',
          f'The average cumulative reward is: {cumulative_reward}\n',
          f'The average regret is: {(cumulative_optimal_reward - cumulative_reward)}'
          )

    print("=======================Explore-Then-Exploit algorithm=======================")
    m = 30
    exploration_phase_length = m * num_of_arms      # number of rounds in the exploration phase

    # Initialize arrays to store the number of pulls and rewards for each arm
    pull_counts = np.zeros(num_of_arms, dtype=int)
    rewards = np.zeros(num_of_arms, dtype=float)

    # reward in each round average by # of iteration
    reward_round_iteration = np.zeros((T), dtype=int)

    # Initialize the best arm as the first arm
    current_best_arm = 0
    current_best_reward = cal_uni_expectation(winning_parameters[current_best_arm][1], winning_parameters[current_best_arm][0])

    for iteration_count in range(total_iteration):
        exploration_phase = True
        for round in range(T):
            # select the best arm with a specific algorithm
            select_arm = arm_selection_ete_policy(round, num_of_arms, exploration_phase, current_best_arm, pull_counts, rewards)
            # generate reward for the selected arm
            reward = cal_uni_expectation(winning_parameters[select_arm][1], winning_parameters[select_arm][0])
            reward_round_iteration[round] += reward
            rewards[select_arm] += reward
            if pull_counts[select_arm] == 1:
                # Update the best arm after each arm has been tried once
                if reward > current_best_reward:
                    current_best_arm = select_arm
                    current_best_reward = reward
            if round >= exploration_phase_length - 1:
                exploration_phase = False

    # compute average reward for each round
    average_reward_in_each_round = np.zeros(T, dtype=float)
    for round in range(T):
        average_reward_in_each_round[round] = float(reward_round_iteration[round])/float(total_iteration)
    
    # Let generate X and Y data points to plot it out
    cumulative_optimal_reward = 0.0
    cumulative_reward = 0.0
    Y_ete_30 = np.zeros(T, dtype=float)
    Y2_ete_30 = np.zeros(T, dtype=float)
    for round in range(T):
        cumulative_optimal_reward += cal_uni_expectation(winning_parameters[optimal_arm][1], winning_parameters[optimal_arm][0])
        cumulative_reward += average_reward_in_each_round[round]
        Y2_ete_30[round] = cumulative_reward
        Y_ete_30[round] = (cumulative_optimal_reward - cumulative_reward)

    print(f'After {T} rounds:\n',
          f'The average cumulative reward is: {cumulative_reward}\n',
          f'The average regret is: {(cumulative_optimal_reward - cumulative_reward)}'
          )
    
    print("=======================Explore-Then-Exploit algorithm=======================")
    m = 40
    exploration_phase_length = m * num_of_arms      # number of rounds in the exploration phase

    # Initialize arrays to store the number of pulls and rewards for each arm
    pull_counts = np.zeros(num_of_arms, dtype=int)
    rewards = np.zeros(num_of_arms, dtype=float)

    # reward in each round average by # of iteration
    reward_round_iteration = np.zeros((T), dtype=int)

    # Initialize the best arm as the first arm
    current_best_arm = 0
    current_best_reward = cal_uni_expectation(winning_parameters[current_best_arm][1], winning_parameters[current_best_arm][0])

    for iteration_count in range(total_iteration):
        exploration_phase = True
        for round in range(T):
            # select the best arm with a specific algorithm
            select_arm = arm_selection_ete_policy(round, num_of_arms, exploration_phase, current_best_arm, pull_counts, rewards)
            # generate reward for the selected arm
            reward = cal_uni_expectation(winning_parameters[select_arm][1], winning_parameters[select_arm][0])
            reward_round_iteration[round] += reward
            rewards[select_arm] += reward
            if pull_counts[select_arm] == 1:
                # Update the best arm after each arm has been tried once
                if reward > current_best_reward:
                    current_best_arm = select_arm
                    current_best_reward = reward
            if round >= exploration_phase_length - 1:
                exploration_phase = False

    # compute average reward for each round
    average_reward_in_each_round = np.zeros(T, dtype=float)
    for round in range(T):
        average_reward_in_each_round[round] = float(reward_round_iteration[round])/float(total_iteration)
    
    # Let generate X and Y data points to plot it out
    cumulative_optimal_reward = 0.0
    cumulative_reward = 0.0
    Y_ete_40 = np.zeros(T, dtype=float)
    Y2_ete_40 = np.zeros(T, dtype=float)
    for round in range(T):
        cumulative_optimal_reward += cal_uni_expectation(winning_parameters[optimal_arm][1], winning_parameters[optimal_arm][0])
        cumulative_reward += average_reward_in_each_round[round]
        Y2_ete_40[round] = cumulative_reward
        Y_ete_40[round] = (cumulative_optimal_reward - cumulative_reward)

    print(f'After {T} rounds:\n',
          f'The average cumulative reward is: {cumulative_reward}\n',
          f'The average regret is: {(cumulative_optimal_reward - cumulative_reward)}'
          )
    
    print("=======================Explore-Then-Exploit algorithm=======================")
    m = 50
    exploration_phase_length = m * num_of_arms      # number of rounds in the exploration phase

    # Initialize arrays to store the number of pulls and rewards for each arm
    pull_counts = np.zeros(num_of_arms, dtype=int)
    rewards = np.zeros(num_of_arms, dtype=float)

    # reward in each round average by # of iteration
    reward_round_iteration = np.zeros((T), dtype=int)

    # Initialize the best arm as the first arm
    current_best_arm = 0
    current_best_reward = cal_uni_expectation(winning_parameters[current_best_arm][1], winning_parameters[current_best_arm][0])

    for iteration_count in range(total_iteration):
        exploration_phase = True
        for round in range(T):
            # select the best arm with a specific algorithm
            select_arm = arm_selection_ete_policy(round, num_of_arms, exploration_phase, current_best_arm, pull_counts, rewards)
            # generate reward for the selected arm
            reward = cal_uni_expectation(winning_parameters[select_arm][1], winning_parameters[select_arm][0])
            reward_round_iteration[round] += reward
            rewards[select_arm] += reward
            if pull_counts[select_arm] == 1:
                # Update the best arm after each arm has been tried once
                if reward > current_best_reward:
                    current_best_arm = select_arm
                    current_best_reward = reward
            if round >= exploration_phase_length - 1:
                exploration_phase = False

    # compute average reward for each round
    average_reward_in_each_round = np.zeros(T, dtype=float)
    for round in range(T):
        average_reward_in_each_round[round] = float(reward_round_iteration[round])/float(total_iteration)
    
    # Let generate X and Y data points to plot it out
    cumulative_optimal_reward = 0.0
    cumulative_reward = 0.0
    Y_ete_50 = np.zeros(T, dtype=float)
    Y2_ete_50 = np.zeros(T, dtype=float)
    for round in range(T):
        cumulative_optimal_reward += cal_uni_expectation(winning_parameters[optimal_arm][1], winning_parameters[optimal_arm][0])
        cumulative_reward += average_reward_in_each_round[round]
        Y2_ete_50[round] = cumulative_reward
        Y_ete_50[round] = (cumulative_optimal_reward - cumulative_reward)

    print(f'After {T} rounds:\n',
          f'The average cumulative reward is: {cumulative_reward}\n',
          f'The average regret is: {(cumulative_optimal_reward - cumulative_reward)}'
          )
    
    
    # adaptive greedy
    print(f"=======================Adaptive Epsilon-Greedy algorithm=======================")

    pull_counts = np.zeros(num_of_arms, dtype=int)
    rewards = np.zeros(num_of_arms, dtype=float)
    reward_round_iteration = np.zeros(T, dtype=float)

    for iteration_count in range(total_iteration):
        for round in range(T):
            select_arm = arm_selection_epsilon_greedy_policy(num_of_arms, epsilon_fuc(round), pull_counts, rewards)
            reward = cal_uni_expectation(winning_parameters[select_arm][1], winning_parameters[select_arm][0])
            rewards[select_arm] += reward
            reward_round_iteration[round] += reward
    
    # compute average reward for each round
    average_reward_in_each_round = np.zeros(T, dtype=float)
    for round in range(T):
        average_reward_in_each_round[round] = reward_round_iteration[round] / total_iteration
    
    # Let generate X and Y data points to plot it out
    cumulative_optimal_reward = 0.0
    cumulative_reward = 0.0
    cumulative_avg_reward = 0.0
    Y_adaptive = np.zeros(T, dtype=float)
    Y2_adaptive = np.zeros(T, dtype=float)
    Linear = np.zeros(T, dtype=float)
    for round in range(T):
        cumulative_optimal_reward += cal_uni_expectation(winning_parameters[optimal_arm][1], winning_parameters[optimal_arm][0])
        cumulative_reward += average_reward_in_each_round[round]
        cumulative_avg_reward += reward_avg
        Y2_adaptive[round] = cumulative_reward
        Y_adaptive[round] = cumulative_optimal_reward - cumulative_reward
        Linear[round] += cumulative_optimal_reward - cumulative_avg_reward

    print(f'After {T} rounds:\n',
          f'The average cumulative reward is: {cumulative_reward}\n',
          f'The average regret is: {(cumulative_optimal_reward - cumulative_reward)}'
          )

    # plot
    fig1, ax1 = plt.subplots()
    fig1.suptitle('Cumulative Average Regret per round')

    # 为每个Y轴数据系列绘制线条
    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'teal', 'orange', 'purple', 'brown', 'gray', 'pink', 'gold', 'violet']
    ax1.plot(X, Y_random, color=colors[0], label=f'random policy')
    ax1.plot(X, Y_ete_1, color=colors[1], label=f'ETE m=1 policy')
    ax1.plot(X, Y_ete_10, color=colors[2], label=f'ETE m=10 policy')
    ax1.plot(X, Y_ete_20, color=colors[3], label=f'ETE m=20 policy')
    ax1.plot(X, Y_ete_30, color=colors[4], label=f'ETE m=30 policy')
    ax1.plot(X, Y_ete_40, color=colors[5], label=f'ETE m=40 policy')
    ax1.plot(X, Y_ete_50, color=colors[6], label=f'ETE m=50 policy')
    ax1.plot(X, Y_greedy_01, color=colors[7], label=f'basic epsilon=0.1 greedy policy')
    ax1.plot(X, Y_greedy_02, color=colors[8], label=f'basic epsilon=0.2 greedy policy')
    ax1.plot(X, Y_greedy_03, color=colors[9], label=f'basic epsilon=0.3 greedy policy')
    ax1.plot(X, Y_greedy_04, color=colors[10], label=f'basic epsilon=0.4 greedy policy')
    ax1.plot(X, Y_greedy_05, color=colors[11], label=f'basic epsilon=0.5 greedy policy')
    ax1.plot(X, Y_ucb, color=colors[12], label=f'ucb policy')
    ax1.plot(X, Y_adaptive, color=colors[13], label=f'adaptive epsilon greedy policy')
    ax1.plot(X, Linear, color=colors[14], label=f'"linear" line', linestyle='--')

    ax1.set(xlabel='round number', ylabel='Cumulative Average Regret')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    ax1.set_xlim(0, T)
    ax1.set_ylim(0, 1.1 * max([Y_random[-1], Y_ete_1[-1], Y_ete_10[-1], Y_ete_20[-1], Y_ete_30[-1], Y_ete_40[-1], Y_ete_50[-1], \
                               Y_greedy_01[-1], Y_greedy_02[-1], Y_greedy_03[-1], Y_greedy_04[-1], Y_greedy_05[-1],\
                               Y_ucb[-1], Y_adaptive[-1]]))
    # 保存第一个图
    plt.savefig(f"Cumulativ_Average_Regret_T{T}.png")
    plt.show()
    
    # 创建第二个图，用于显示Cumulative Average Reward
    fig2, ax2 = plt.subplots()
    fig2.suptitle('Cumulative Average Reward per round')
    # 为每个Y2轴数据系列绘制线条
    ax2.plot(X, Y2_random, color=colors[0], label=f'random policy')
    ax2.plot(X, Y2_ete_1, color=colors[1], label=f'ETE m=1 policy')
    ax2.plot(X, Y2_ete_10, color=colors[2], label=f'ETE m=10 policy')
    ax2.plot(X, Y2_ete_20, color=colors[3], label=f'ETE m=20 policy')
    ax2.plot(X, Y2_ete_30, color=colors[4], label=f'ETE m=30 policy')
    ax2.plot(X, Y2_ete_40, color=colors[5], label=f'ETE m=40 policy')
    ax2.plot(X, Y2_ete_50, color=colors[6], label=f'ETE m=50 policy')
    ax2.plot(X, Y2_greedy_01, color=colors[7], label=f'basic epsilon=0.1 greedy policy')
    ax2.plot(X, Y2_greedy_02, color=colors[8], label=f'basic epsilon=0.2 greedy policy')
    ax2.plot(X, Y2_greedy_03, color=colors[9], label=f'basic epsilon=0.3 greedy policy')
    ax2.plot(X, Y2_greedy_04, color=colors[10], label=f'basic epsilon=0.4 greedy policy')
    ax2.plot(X, Y2_greedy_05, color=colors[11], label=f'basic epsilon=0.5 greedy policy')
    ax2.plot(X, Y2_ucb, color=colors[12], label=f'ucb policy')
    ax2.plot(X, Y2_adaptive, color=colors[13], label=f'adaptive epsilon greedy policy')
    ax2.set(xlabel='round number', ylabel='Cumulative Average Reward')
    ax2.grid(True)
    ax2.legend(loc='upper left')
    ax2.set_xlim(0, T)
    ax2.set_ylim(0, 1.1 * max([Y2_random[-1], Y2_ete_1[-1], Y2_ete_10[-1], Y2_ete_20[-1], Y2_ete_30[-1], Y2_ete_40[-1], Y2_ete_50[-1], \
                               Y2_greedy_01[-1], Y2_greedy_02[-1], Y2_greedy_03[-1], Y2_greedy_04[-1], Y2_greedy_05[-1],\
                               Y2_ucb[-1], Y2_adaptive[-1]]))
    # 保存第二个图
    plt.savefig(f"Cumulative_Average_Reward_T{T}.png")
    plt.show()