"""
CMSC5728 Programming Assignment #1
Author: HUANG, Hao Yu
Date:   10/10/2024
"""
import random
import numpy as np
import math
import matplotlib.pyplot as plt


def generate_random_outcome(lowbound1, upbound1, lowbound2, upbound2):
    """
    Target:     Determine choose random from which range everytiem I generate a number.
    """

    if random.choice([True, False]):
        return random.uniform(lowbound1, upbound1)
    else:
        return random.uniform(lowbound2, upbound2)

def uni_distribution(path, lowbound, upbound, num, if_regen=False):
    """
    Target: 1.  Generate 50 random outcomes from a uniform distribution between [lowbound, upbound],
                store these 50 numbers in a file, say input.
            2.  Calculate avg of this distribution
    Return: Avg of this distribution
    """

    avg_uni = float(1 / (upbound - lowbound))
    if if_regen:
        print("Begin Generating and Writing Variables...")
        random_outcomes = [random.uniform(lowbound, upbound) for _ in range(num)]
        with open(path, 'w') as f:
            for item in random_outcomes:
                f.write(str(item) + '\n')
        print(f"Sample Data Generated Variables: {random_outcomes}")
    else:
        print("Already Generated.")
    print("Done")
    print("")
    return avg_uni

def uni_distribution_2range(path, lowbound1, upbound1, lowbound2, upbound2, num, if_regen=False):
    """
    Target: 1.  Generate 50 random outcomes from a uniform distribution between [lowbound, upbound],
                store these 50 numbers in a file, say input.
            2.  Calculate avg of this distribution
    Return: Avg of this distribution
    """

    avg_uni = 1/2 * float(1 / (upbound1 - lowbound1)) + 1/2 * float(1 / (upbound2 - lowbound2))
    if if_regen:
        print("Begin Generating and Writing Variables...")
        random_outcomes = [generate_random_outcome(lowbound1, upbound1, lowbound2, upbound2) for _ in range(num)]
        with open(path, 'w') as f:
            for item in random_outcomes:
                f.write(str(item) + '\n')
        print(f"Sample Data Generated Variables: {random_outcomes}")
    else:
        print("Already Generated.")
    print("Done")
    print("")
    return avg_uni

def calculate_empirical_avg(input_file):
    """
    Target:     For this probability distribution, calculate the average value, call this the 
                empirical average.
    Return:     Empirical Average, var list
    """

    print("Begin Calculating Empirical Average of Variables...")
    var_list = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            var_list.append(float(line.strip('\n')))
    var_list = np.array(var_list)
    e_avg_uni = np.sum(var_list) / var_list.shape[0]
    print(f"Empirical Average: {e_avg_uni}")
    print("Done")
    print("")
    return e_avg_uni, var_list

def calculate_hoeffding_confidence_interval(e_avg_uni, var_list, biggest_interval, confidence_level):
    """
    Target:     Write a program which can use  the Hoeffding's Inequality to compute the 
                confidence interval of the average value for each of these 50 samples. In other 
                words, find  the upper/lower bounds after i samples, where 𝑖∈{1,2,...,50}.
    return:     The lower bound and upper bound of the true distribution average.
    """
    
    print(f"Begin Calcualting Hoeffding Confidence Inteval With Confidence: {(1 - confidence_level)*100}%...")
    n = var_list.shape[0]
    epsilon = math.sqrt(biggest_interval**2 * math.log(2 / confidence_level) / (2 * n))
    real_lower_bound = e_avg_uni - epsilon
    real_upper_bound = e_avg_uni + epsilon
    print(f"Lower Bound:{real_lower_bound} \nUpper Bound:{real_upper_bound}")
    print("Done")
    print("")
    return real_lower_bound, real_upper_bound


if __name__ == "__main__":
    confidence_level = 0.05
    input_path1 = './input1.txt'
    input_path2 = './input2.txt'
    input_path3 = './input3.txt'
    # case1
    print("===============================Case1: Uniform Distribution [0,1]==============================\n")
    avg_uni = uni_distribution(input_path1, 0, 1, 50, if_regen=False)
    e_avg_uni, var_list = calculate_empirical_avg(input_path1)
    biggest_interval = 1 - 0
    # with 95% confidence
    real_lower_bound, real_upper_bound = calculate_hoeffding_confidence_interval(e_avg_uni, var_list, biggest_interval, confidence_level=confidence_level)
    print("==========================================Case1 Done==========================================\n")
    
    # case2
    print("===============================Case2: Uniform Distribution [0,2]==============================\n")
    avg_uni = uni_distribution(input_path2, 0, 2, 50, if_regen=False)
    e_avg_uni, var_list = calculate_empirical_avg(input_path2)
    biggest_interval = 2 - 0
    # with 95% confidence
    real_lower_bound, real_upper_bound = calculate_hoeffding_confidence_interval(e_avg_uni, var_list, biggest_interval, confidence_level=confidence_level)
    print("==========================================Case2 Done==========================================\n")
    
    # case 3
    print("============================Case3: Uniform Distribution [0,1] and [3,4]=========================\n")
    avg_uni = uni_distribution_2range(input_path3, 0, 1, 3, 4, 50, if_regen=False)
    e_avg_uni, var_list = calculate_empirical_avg(input_path3)
    biggest_interval = 4 - 0
    # with 95% confidence
    real_lower_bound, real_upper_bound = calculate_hoeffding_confidence_interval(e_avg_uni, var_list, biggest_interval, confidence_level=confidence_level)
    print("==========================================Case3 Done==========================================\n")
