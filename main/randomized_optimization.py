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
import mlrose
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
  
    problem3 = mlrose.DiscreteOpt(length=N, fitness_fn=fitness_cust3, maximize=True, max_val=2)
    comparator3 = RandomOptimizerComparator(problem3, init_state, "Modular Sums", image_path=analysis_dir)
    comparator3.run_analysis()
    time.sleep(5)
    
    problem9 = mlrose.DiscreteOpt(length=N, fitness_fn=fitness_cust9, maximize=True, max_val=2)
    comparator9 = RandomOptimizerComparator(problem9, init_state, "Magic Triplets", image_path=analysis_dir)
    comparator9.run_analysis()
    time.sleep(5)

    four_peaks_problem = mlrose.DiscreteOpt(length=N, fitness_fn=mlrose.FourPeaks(), maximize=True, max_val=2)
    four_peaks_comparator = RandomOptimizerComparator(four_peaks_problem, init_state, "Four Peaks", image_path=analysis_dir)
    four_peaks_comparator.run_analysis()


def queens_example():
    print("Start Queens Example....")  
    
    fitness = mlrose.Queens()  
    problem = mlrose.DiscreteOpt(length=8, fitness_fn=fitness, maximize=False, max_val=8)
    
    # Define decay schedule
    schedule = mlrose.ExpDecay()

    # Define initial state
    init_state = np.array([1, 0, 0, 0, 0, 0, 0, 0])

    # Solve problem using simulated annealing
    best_state, best_fitness = mlrose.simulated_annealing(problem, schedule=schedule,
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
fitness_cust_queens_max = mlrose.CustomFitness(queens_max)


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
fitness_cust1 = mlrose.CustomFitness(prob1_max)


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
fitness_cust2 = mlrose.CustomFitness(prob2_max)


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
fitness_cust3 = mlrose.CustomFitness(prob3_max)


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
fitness_cust4 = mlrose.CustomFitness(prob4_max)


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
fitness_cust5 = mlrose.CustomFitness(prob5_max)

    
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
fitness_cust6 = mlrose.CustomFitness(prob6_max)


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
fitness_cust7 = mlrose.CustomFitness(prob7_max)


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
fitness_cust8 = mlrose.CustomFitness(prob8_max)


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
fitness_cust9 = mlrose.CustomFitness(prob9_max)


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
fitness_cust10 = mlrose.CustomFitness(prob10_max)

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
