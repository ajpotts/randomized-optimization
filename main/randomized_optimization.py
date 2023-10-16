#!/usr/bin/env python
# encoding: utf-8
'''
main.randomized_optimization -- shortdesc

main.randomized_optimization is a description

It defines classes_and_methods

@author:     user_name

@copyright:  2023 organization_name. All rights reserved.

@license:    license

@contact:    user_email
@deffield    updated: Updated
'''

import os
import sys
import six
import math
import time
import pathlib
from pathlib import Path

import base64
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

sys.modules['sklearn.externals.six'] = six
import joblib
import mlrose_hiive
import numpy as np
from randomized_optimizer_comparator import RandomOptimizerComparator

__all__ = []
__version__ = 0.1
__date__ = '2023-10-10'
__updated__ = '2023-10-10'

DEBUG = 1
TESTRUN = 0
PROFILE = 0


def main(argv=None):

    np.random.seed(0)
    N = 100
    init_state = np.random.randint(2, size=N)  # np.zeros(N)
    
    path = pathlib.Path().resolve()
    project_dir = str(Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute())
    analysis_dir = str(path) + '/analysis/'

    one_max_problem = mlrose_hiive.DiscreteOpt(length=N, fitness_fn=mlrose_hiive.OneMax(), maximize=True, max_val=2)
    one_max_comparator = RandomOptimizerComparator(one_max_problem, init_state, "one_max")
    one_max_comparator.run_analysis()

    flip_flop_problem = mlrose_hiive.DiscreteOpt(length=N, fitness_fn=mlrose_hiive.FlipFlop(), maximize=True, max_val=2)
    flip_flop_comparator = RandomOptimizerComparator(flip_flop_problem, init_state, "flip_flop")
    flip_flop_comparator.run_analysis()
        
    cts_peaks_problem = mlrose_hiive.DiscreteOpt(length=N, fitness_fn=mlrose_hiive.ContinuousPeaks(t_pct=0.15), maximize=True, max_val=2)
    cts_peaks_comparator = RandomOptimizerComparator(cts_peaks_problem, init_state, "Continuous Peaks")
    cts_peaks_comparator.run_analysis()        

    queens_problem = mlrose_hiive.DiscreteOpt(length=8, fitness_fn=fitness_cust_queens_max, maximize=True, max_val=8) 
    queens_comparator = RandomOptimizerComparator(queens_problem, [1, 3, 2, 4, 3, 5, 4, 6], "8 Queens")
    queens_comparator.run_analysis()    
    
    problem1 = mlrose_hiive.DiscreteOpt(length=N, fitness_fn=fitness_cust1, maximize=True, max_val=2)
    comparator1 = RandomOptimizerComparator(problem1, init_state, "Problem1")
    comparator1.run_analysis()    
    
    problem2 = mlrose_hiive.DiscreteOpt(length=N, fitness_fn=fitness_cust2, maximize=True, max_val=2)
    comparator2 = RandomOptimizerComparator(problem2, init_state, "Palindromes")
    comparator2.run_analysis()
    
    problem4 = mlrose_hiive.DiscreteOpt(length=N, fitness_fn=fitness_cust4, maximize=True, max_val=2)
    comparator4 = RandomOptimizerComparator(problem4, init_state, "Average Run Length")
    comparator4.run_analysis()
    
    problem5 = mlrose_hiive.DiscreteOpt(length=N, fitness_fn=fitness_cust5, maximize=True, max_val=2)
    comparator5 = RandomOptimizerComparator(problem5, init_state, "Problem 5")
    comparator5.run_analysis()
    
    key = get_key()
    print(key[0:50])

    problem6 = mlrose_hiive.DiscreteOpt(length=N, fitness_fn=fitness_cust6, maximize=True, max_val=2)
    comparator6 = RandomOptimizerComparator(problem6, init_state, "Key Guessing")
    comparator6.run_analysis()
    
    problem7 = mlrose_hiive.DiscreteOpt(length=N, fitness_fn=fitness_cust7, maximize=True, max_val=2)
    comparator7 = RandomOptimizerComparator(problem7, init_state, "Random Weights")
    comparator7.run_analysis()
    
    problem8 = mlrose_hiive.DiscreteOpt(length=N, fitness_fn=fitness_cust8, maximize=True, max_val=2)
    comparator8 = RandomOptimizerComparator(problem8, init_state, "Unique Integers")
    comparator8.run_analysis()

    problem10 = mlrose_hiive.DiscreteOpt(length=N, fitness_fn=fitness_cust10, maximize=True, max_val=2)
    comparator10 = RandomOptimizerComparator(problem10, init_state, "Two Peaks")
    comparator10.run_analysis()  
    
        # weights = [10, 5, 2, 8, 15]
    # values = [1, 2, 3, 4, 5]
    # max_weight_pct = 0.6
    # Knapsack_problem = mlrose_hiive.DiscreteOpt(length=N, fitness_fn=mlrose_hiive.Knapsack(weights, values, max_weight_pct), maximize=True, max_val=2)
    # Knapsack_comparator = RandomOptimizerComparator(Knapsack_problem, init_state, "Knapsack Problem")
    # Knapsack_comparator.run_analysis()
    #

    # # Create list of city coordinates
    # coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]
    #
    # # Initialize fitness function object using coords_list
    # fitness_coords = mlrose_hiive.TravellingSales(coords = coords_list)
    #
    # # Create list of distances between pairs of cities
    # dist_list = [(0, 1, 3.1623), (0, 2, 4.1231), (0, 3, 5.8310), (0, 4, 4.2426), \
    #          (0, 5, 5.3852), (0, 6, 4.0000), (0, 7, 2.2361), (1, 2, 1.0000), \
    #          (1, 3, 2.8284), (1, 4, 2.0000), (1, 5, 4.1231), (1, 6, 4.2426), \
    #          (1, 7, 2.2361), (2, 3, 2.2361), (2, 4, 2.2361), (2, 5, 4.4721), \
    #          (2, 6, 5.0000), (2, 7, 3.1623), (3, 4, 2.0000), (3, 5, 3.6056), \
    #          (3, 6, 5.0990), (3, 7, 4.1231), (4, 5, 2.2361), (4, 6, 3.1623), \
    #          (4, 7, 2.2361), (5, 6, 2.2361), (5, 7, 3.1623), (6, 7, 2.2361)]
    #
    # # Initialize fitness function object using dist_list
    # fitness_dists = mlrose_hiive.TravellingSales(distances = dist_list)
    #
    #
    # traveling_sales_problem = mlrose_hiive.TSPOpt(length = 8, fitness_fn = fitness_dists, maximize=False)
    # traveling_sales_comparator = RandomOptimizerComparator(traveling_sales_problem, [0,1,2,3,4,5,6,7], "Traveling Salesperson")
    # traveling_sales_comparator.run_analysis()
    
    ##################################################3
  
    problem3 = mlrose_hiive.DiscreteOpt(length=N, fitness_fn=fitness_cust3, maximize=True, max_val=2)
    comparator3 = RandomOptimizerComparator(problem3, init_state, "Modular Sums", image_path=analysis_dir)
    comparator3.run_analysis()
    time.sleep(5)
    
    problem9 = mlrose_hiive.DiscreteOpt(length=N, fitness_fn=fitness_cust9, maximize=True, max_val=2)
    comparator9 = RandomOptimizerComparator(problem9, init_state, "Magic Triplets", image_path=analysis_dir)
    comparator9.run_analysis()
    time.sleep(5)

    four_peaks_problem = mlrose_hiive.DiscreteOpt(length=N, fitness_fn=mlrose_hiive.FourPeaks(), maximize=True, max_val=2)
    four_peaks_comparator = RandomOptimizerComparator(four_peaks_problem, init_state, "Four Peaks", image_path=analysis_dir)
    four_peaks_comparator.run_analysis()


def queens_example():
    print("Start Queens Example....")  
    
    fitness = mlrose_hiive.Queens()  
    problem = mlrose_hiive.DiscreteOpt(length=8, fitness_fn=fitness, maximize=False, max_val=8)
    
    # Define decay schedule
    schedule = mlrose_hiive.ExpDecay()

    # Define initial state
    init_state = np.array([1, 0, 0, 0, 0, 0, 0, 0])

    # Solve problem using simulated annealing
    best_state, best_fitness = mlrose_hiive.simulated_annealing(problem, schedule=schedule,
                                                      max_attempts=100, max_iters=1000,
                                                      init_state=init_state, random_state=1)
    
    print("End Queens Example.")


# Define alternative N-Queens fitness function for maximization problem
def queens_max(state):

    # Initialize counter
    fitness_cnt = 0

    # For all pairs of queens
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):

            # Check for horizontal, diagonal-up and diagonal-down attacks
            if (state[j] != state[i]) \
                and (state[j] != state[i] + (j - i)) \
                and (state[j] != state[i] - (j - i)):

                # If no attacks, then increment counter
                fitness_cnt += 1

    return fitness_cnt


# Initialize custom fitness function object
fitness_cust_queens_max = mlrose_hiive.CustomFitness(queens_max)


def prob1_max(state, k=10):

    # Initialize counter
    fitness_cnt = 0
    
    # For all pairs of queens
    for i in range(1, len(state) - 1):
        
        if(state[i] != state[i - 1]):
            fitness_cnt += 1
    
    fitness = -1 * fitness_cnt * fitness_cnt + 2 * k * fitness_cnt

    return fitness


# Initialize custom fitness function object
fitness_cust1 = mlrose_hiive.CustomFitness(prob1_max)


def prob2_max(state):

    # Initialize counter
    fitness_cnt = 0

    L = len(state)
    r = int(L / 2)
    
    # For all pairs of queens
    for i in range(r):
        
        if(state[i] == state[L - 1 - i]):
            fitness_cnt += 1
    
    sum = 0       
    for i in range(L - 1):
       sum += state[i]
       
    if(sum == 0):
        fitness_cnt = 0 

    return fitness_cnt


# Initialize custom fitness function object
fitness_cust2 = mlrose_hiive.CustomFitness(prob2_max)


def prob3_max(state, k=6, m=3, r=1):

    # Initialize counter
    fitness_cnt = 0

    # For all pairs of queens
    for i in range(len(state) - 1):

        sum = 0

        for j in range(0, k):
            if(i > j):
                sum += state[i - j]

        if(sum % m == r):

            fitness_cnt += 1

    return fitness_cnt


# Initialize custom fitness function object
fitness_cust3 = mlrose_hiive.CustomFitness(prob3_max)


def prob4_max(state, k=20):
    
    runs = []    
    run_len = 1
    
    # For all pairs of queens
    for i in range(1, len(state) - 1):
        
        if state[i] == state[i - 1]:
            run_len += 1
        
        if(state[i] != state[i - 1]):
            runs.append(run_len)
            run_len = 1
    
    runs.append(run_len)
            
    avg_runs = sum(runs) / len(runs)
    
    fitness = -1 * avg_runs * avg_runs + 2 * k * avg_runs

    return fitness


# Initialize custom fitness function object
fitness_cust4 = mlrose_hiive.CustomFitness(prob4_max)


def prob5_max(state, k=4):
    
    fitness = 0
    
    one_runs = []    
    zero_runs = []
    run_len = 1
    
    L = len(state) - 1
    
    # For all pairs of queens
    for i in range(1, L):
        
        if state[i] == state[i - 1]:
            run_len += 1
        
        if(state[i] != state[i - 1] and run_len % k == 0):
            if(state[i] == 1):
                one_runs.append(run_len)
                if(run_len % (k * k) == 0):
                    fitness += 10
                
            elif(run_len % k == 0):
                zero_runs.append(run_len)                
            run_len = 1
    
    if(state[L] == 1):
        one_runs.append(run_len)
    else:
        zero_runs.append(run_len)
    
    num_one_runs = len(one_runs)
 
    max_one_run = 0
    max_zero_run = 0
 
    if(num_one_runs > 0): 
        avg_one_runs = sum(one_runs) / num_one_runs
        max_one_run = max(one_runs)
    else:
        avg_one_runs = 0
        
    num_zero_runs = len(zero_runs)
 
    if(num_zero_runs > 0): 
        avg_zero_runs = sum(zero_runs) / num_zero_runs
        max_zero_run = max(zero_runs)
    else:
        avg_zero_runs = 0        
    
    # fitness = num_one_runs

    fitness += num_one_runs % k
    fitness += num_zero_runs % k
    fitness += avg_one_runs % k
    fitness += avg_zero_runs % k
    fitness += max_one_run
    fitness += max_zero_run
    
    # x = num_one_runs - num_zero_runs
    #
    # fitness += -1 * x * x + 2 * k * x
    
    if(num_one_runs % (k * k) == 0):
        fitness += 10

    if(num_one_runs % (k * k) == 0):
        fitness += 10

    return fitness


# Initialize custom fitness function object
fitness_cust5 = mlrose_hiive.CustomFitness(prob5_max)

    
def get_key():
    
    password_provided = "password"  # This is input in the form of a string
    password = password_provided.encode()  # Convert to type bytes
    salt = b'salt_'  # CHANGE THIS - recommend using a key from os.urandom(16), must be of type bytes
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = base64.urlsafe_b64encode(kdf.derive(password))  # Can only use kdf once
    
    # key = Fernet.generate_key()

    int_val = int.from_bytes(key, "big")
    
    b = bin(int_val)
    c = [x for x in b][3:]
    c = [int(x) for x in c]
    
    return c


def prob6_max(state):
    
    fitness = 0
    
    key = get_key()
    
    # For all pairs of queens
    for i in range(0, len(state) - 1):
        
        if state[i] == key[i]:
            fitness += 1

    return fitness


# Initialize custom fitness function object
fitness_cust6 = mlrose_hiive.CustomFitness(prob6_max)


def prob7_max(state):
    
    fitness = 0
    
    L = len (state)
    
    weight = np.random.randint(10, size=N)
    exponent = np.random.randint(2, size=N)
    
    # For all pairs of queens
    for i in range(0, len(state) - 1):

        fitness += state[i] * weight[i] * (-1) ** exponent[i]

    return fitness


# Initialize custom fitness function object
fitness_cust7 = mlrose_hiive.CustomFitness(prob7_max)


def prob8_max(state):
    
    fitness = 0
    
    L = len (state)
    
    N = int(L / 10)
    
    weight = np.random.randint(10, size=N)
    exponent = np.random.randint(2, size=N)
    
    int_set = set()
    
    # For all pairs of queens
    for j in range(10):
        current_int = 0
        for i in range(0, N - 1):
            current_int += 2 ** i * state[8 * j + i]
        int_set.add(current_int)
    
    fitness = len(int_set)
    
    return fitness


# Initialize custom fitness function object
fitness_cust8 = mlrose_hiive.CustomFitness(prob8_max)


def prob9_max(state):
    
    np.random.seed(0)
    
    fitness = 0
    
    L = len (state)
    
    triplets1 = []
    triplets0 = []
    
    N = int(L / 10)
    
    for i in range(N):
        triplets1.append(np.random.randint(L, size=N))
        triplets1.append(np.random.randint(L, size=N))
    
    for triplet in triplets1:

        fitness += state[triplet[0]] * state[triplet[1]] * state[triplet[2]]
        
    for triplet in triplets0:

        fitness += (1 - state[triplet[0]]) * (1 - state[triplet[1]]) * (1 - state[triplet[2]])  
    
    return fitness


# Initialize custom fitness function object
fitness_cust9 = mlrose_hiive.CustomFitness(prob9_max)


def prob10_max(state):
    
    L = len (state)
    
    f1 = 0
    f2 = 0
    
    for i in range(L):
        if(state[i]) == 0:
            break
        else:
            f1 += 1
        
    for i in range(L):
        if(state[L - 1 - i]) == 1:
            break
        else:
            f1 += 1    
    
    fitness = math.sqrt(f1) + math.sqrt(f1)
    
    return fitness


# Initialize custom fitness function object
fitness_cust10 = mlrose_hiive.CustomFitness(prob10_max)

if __name__ == "__main__":
    if DEBUG:
        sys.argv.append("-h")
    if TESTRUN:
        import doctest
        doctest.testmod()
    if PROFILE:
        import cProfile
        import pstats
        profile_filename = 'main.randomized_optimization_profile.txt'
        cProfile.run('main()', profile_filename)
        statsfile = open("profile_stats.txt", "wb")
        p = pstats.Stats(profile_filename, stream=statsfile)
        stats = p.strip_dirs().sort_stats('cumulative')
        stats.print_stats()
        statsfile.close()
        sys.exit(0)
    sys.exit(main())
