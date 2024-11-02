import random
import numpy as np
import matplotlib.pyplot as plt

def cal_uni_expectation(upper, lower):
    return (upper + lower) / 2

def arm_selection_epsilon_greedy_policy(num_of_arms, epsilon, pull_counts, rewards):
    if random.random() < epsilon:
        select_arm = np.random.randint(num_of_arms)
    else:
        select_arm = np.argmax(rewards / (pull_counts + 1e-10))
    pull_counts[select_arm] += 1
    return select_arm

def epsilon(t):
    # Define epsilon as a function of time t
    return 1.0 / (1.0 + t)


if __name__ == "__main__":
    print(f"=======================Adaptive Epsilon-Greedy algorithm=======================")
    num_of_arms = 10
    winning_parameters = np.array([tuple([0,2]), tuple([1,3]), tuple([2,4]), tuple([3,9]), tuple([4,6]), tuple([8,10]), tuple([3,5]), tuple([4,10]), tuple([5,7]), tuple([6,8])])
    optimal_arm = 5
    T = 10000
    total_iteration = 200

    pull_counts = np.zeros(num_of_arms, dtype=int)
    rewards = np.zeros(num_of_arms, dtype=float)
    reward_round_iteration = np.zeros(T, dtype=float)

    for iteration_count in range(total_iteration):
        for round in range(T):
            select_arm = arm_selection_epsilon_greedy_policy(num_of_arms, epsilon(round), pull_counts, rewards)
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
    X = np.zeros(T, dtype=int)
    Y = np.zeros(T, dtype=float)
    Y2 = np.zeros(T, dtype=float)
    for round in range(T):
        X[round] = round
        cumulative_optimal_reward += cal_uni_expectation(winning_parameters[optimal_arm][1], winning_parameters[optimal_arm][0])
        cumulative_reward += average_reward_in_each_round[round]
        Y2[round] = cumulative_reward
        Y[round] = cumulative_optimal_reward - cumulative_reward

    print(f'After {T} rounds:\n',
          f'The average cumulative reward is: {cumulative_reward}\n',
          f'The average regret is: {(cumulative_optimal_reward - cumulative_reward)}'
          )

    fig, axs = plt.subplots(2)   # get two figures, top is regret, bottom is average reward in each round
    fig.suptitle('Performance of Adaptive Epsilon-Greedy Arm Selection')
    fig.subplots_adjust(hspace=0.5)
    axs[0].plot(X, Y, color='red', label='Regret of Adaptive Epsilon-Greedy Arm Selection Policy')
    axs[0].set(xlabel='round number', ylabel='Regret')
    axs[0].grid(True)
    axs[0].legend(loc='upper left')
    axs[0].set_xlim(0, T)
    axs[0].set_ylim(0, 1.1 * (cumulative_optimal_reward - cumulative_reward))
    axs[1].plot(X, Y2, color='black', label='cumulative average reward')
    axs[1].set(xlabel='round number', ylabel='Cumulative Average Reward per round')
    axs[1].grid(True)
    axs[1].legend(loc='upper left')
    axs[1].set_xlim(0, T)
    axs[1].set_ylim(0, max(Y2))
    plt.savefig("adaptive_epsilon_greedy.png")
    plt.show()