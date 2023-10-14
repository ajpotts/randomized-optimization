'''
Created on Oct 10, 2023

@author: amandapotts
'''

import logging
from pickle import NONE
import six
import threading
import time
import sys
from datetime import date

from mlrose import fitness
import mlrose

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.modules['sklearn.externals.six'] = six


class RandomOptimizerComparator(object):
    '''
    classdocs
    '''

    def __init__(self, problem, init_state, name,
                  verbose=False,
                  image_path="analysis/",
                 population_sizes=[  200, 300, 400, 500, 1000],
                 mutation_probs=[0.0001, 0.001, 0.01, 0.05, 0.1 ],
                 geom_decay=[0.9, 0.99, 0.999, 0.9999, 0.99999],
                 arith_decay=[0.01, 0.001, 0.0001],
                 exp_decay=[0.001, 0.005, 0.05],
                 restarts=10,
                 random_state=1
                 ):
        '''
        Constructor
        '''

        self.problem = problem
        self.init_state = init_state
        self.max_iterations = 5000
        self.max_attempts = 100
        self.random_state = random_state
        self.curve = True
        self.name = name
        self.verbose = verbose
        self.image_path = image_path
        self.population_sizes = population_sizes
        self.mutation_probs = mutation_probs
        self.restarts = restarts
        self.geom_decay = geom_decay
        self.arith_decay = arith_decay
        self.exp_decay = exp_decay
        
        self.threads = []
   
        today = date.today()
        date_string = today.strftime("%Y-%m-%d-%S")    
        logging.basicConfig(filename=self.image_path + "model_run_log_" + date_string + ".txt", level=logging.INFO)

        logging.info('\n\nStarting....\n\n')        
        logging.info('init_state : ' + str(self.init_state))
        logging.info('self.max_iterations : ' + str(self.max_iterations))
        logging.info('self.max_attempts : ' + str(self.max_attempts))                
        logging.info('self.random_state : ' + str(self.random_state))                     
        logging.info('self.name : ' + str(self.name))               
        logging.info('self.image_path : ' + str(self.image_path))
        logging.info('self.population_sizes : ' + str(self.population_sizes))        
        logging.info('self.mutation_probs : ' + str(self.mutation_probs))        
        logging.info('self.restarts : ' + str(self.restarts))  
        logging.info('self.geom_decay : ' + str(self.geom_decay))  
        logging.info('self.arith_decay : ' + str(self.arith_decay))
        logging.info('self.exp_decay : ' + str(self.exp_decay))
        logging.info('self.random_state : ' + str(self.random_state))
        
    def execute_threads(self):
        for thread in self.threads:
            thread.start()
        while self.threads:
            self.threads.pop().join() 
        time.sleep(5)
        
    def run_analysis(self):
        
        logging.info('Problem : ' + self.name)

        hc_results = []
        self.set_hill_climbing_threads(hc_results) 
        
        sa_results = []
        self.set_simulated_annealing_threads(sa_results)
        
        ga_results = []
        self.set_genetic_threads(ga_results, self.population_sizes, self.mutation_probs)
        
        self.execute_threads()
        
        mimic_results = []
        self.set_mimic_threads(mimic_results, self.population_sizes)
        
        self.execute_threads()
        
        best_hill_climbing = self.get_best_result(hc_results)
        curve_hill_climbing = self.get_best_curve(best_hill_climbing)
        hill_climbing_time = self.get_avg_run_time(hc_results)        
        logging.info('Best Hill Climbing : ' + str(best_hill_climbing))
        logging.info('Run Time Hill Climbing : ' + str(hill_climbing_time))
        self.print_fitness_curves(hc_results, "Randomized Hill Climbing", "hill_climbing_params")
                     
        best_annealing = self.get_best_result(sa_results)                                                         
        curve_annealing = self.get_best_curve(best_annealing)
        annealing_time = self.get_avg_run_time(sa_results)
        logging.info('Best Simulated Annealing : ' + str(best_annealing))
        logging.info('Run Time Simulated Annealing : ' + str(annealing_time))
        self.print_fitness_curves(sa_results, "Simulated Annealing", "simulated_annealing_params")

        best_mimic = self.get_best_result(mimic_results)        
        curve_mimic = self.get_best_curve(best_mimic)
        mimic_time = self.get_avg_run_time(mimic_results)
        logging.info('Best MIMIC : ' + str(best_mimic))
        logging.info('Run Time MIMIC : ' + str(mimic_time))
        self.print_fitness_curves(mimic_results, "MIMIC", "mimic_params")
        
        best_genetic = self.get_best_result(ga_results)
        curve_genetic = self.get_best_curve(best_genetic)
        genetic_time = self.get_avg_run_time(ga_results)
        logging.info('Best Genetic Algorithm : ' + str(best_genetic))
        logging.info('Run Time Genetic Algorithm : ' + str(genetic_time))   
        self.print_fitness_curves(ga_results, "Genetic Algorithm", "genetic_alg_params")
        
        blue_patch = mpatches.Patch(color='blue', label="Simulated Annealing")
        green_patch = mpatches.Patch(color='green', label="Randomized Hill Climbing")
        red_patch = mpatches.Patch(color='red', label="MIMIC")
        purple_patch = mpatches.Patch(color='purple', label="Genetic Algorithm")
        
        learning_curve_filename = self.image_path + self.name + "_performance.png"
        runtime_filename = self.image_path + self.name + "_runtime.png"
          
        self.clear_plots()                                   
        ax = plt.gca()
        plt.plot(curve_annealing, color='blue')
        plt.plot(curve_hill_climbing, color='green')       
        plt.plot(curve_mimic, color='red')       
        plt.plot(curve_genetic, color='purple')          
        plt.legend(handles=[blue_patch, green_patch, red_patch, purple_patch], loc="lower right") 
        plt.xlabel("Iterations")
        plt.ylabel("Fitness Score")
        plt.title("Fitness by Number of Iterations for Each Algorithm")
        plt.savefig(learning_curve_filename) 
        if(self.verbose):
            plt.show()
        
        self.clear_plots()  
        ax = plt.gca()
        fig = plt.figure(figsize=(10, 5))
        algorithms = ["Simulated Annealing", "Randomized Hill Climbing", "MIMIC", "Genetic Algorithm"]
        times = [annealing_time, hill_climbing_time, mimic_time, genetic_time]
        
        # creating the bar plot
        plt.bar(algorithms, times, color='maroon',
        width=0.4)
 
        plt.xlabel("Algorithm")
        plt.ylabel("Running Time (Seconds)")
        plt.title("Running Time For Each Algorithm (Seconds)")
        plt.savefig(runtime_filename)
        
        self.get_stats(best_hill_climbing, best_annealing, best_genetic, best_mimic)
        
    def clear_plots(self):
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()
        
    def get_stats(self, best_hill_climbing, best_simulated_annealing, best_genetic, best_mimic):
        
        N = 10
        
        hc_results = []
        self.set_hill_climbing_threads_fixed_param(N, hc_results)
        
        print(best_simulated_annealing)
        print(len(best_simulated_annealing))
        
        sa_results = []
        self.set_simulated_annealing_threads_fixed_params(N, sa_results, str(best_simulated_annealing[4][0]), best_simulated_annealing[4][1])
        
        ga_results = []
        self.set_genetic_threads_fixed_param(ga_results, best_genetic[4][0], best_genetic[4][1])
        
        mimic_results = []
        self.set_mimic_threads_fixed_param(N, mimic_results, best_mimic[4][0])
        self.execute_threads()       
        
        logging.info('Num Runs : ' + str(N))
        logging.info('Randomized Hill Climbing Avg Fitness : ' + str(self.get_avg_fitness(hc_results)))
        logging.info('Randomized Hill Climbing Avg Run Time : ' + str(self.get_avg_run_time(hc_results)))  
            
        logging.info('Simulated Annealing Avg Fitness : ' + str(self.get_avg_fitness(sa_results)))
        logging.info('Simulated Annealing Avg Run Time : ' + str(self.get_avg_run_time(sa_results)))  

        logging.info('Genetic Algorithm Avg Fitness : ' + str(self.get_avg_fitness(ga_results)))
        logging.info('Genetic Algorithm Avg Run Time : ' + str(self.get_avg_run_time(ga_results)))  

        logging.info('MIMIC Avg Fitness : ' + str(self.get_avg_fitness(mimic_results)))
        logging.info('MIMIC Avg Run Time : ' + str(self.get_avg_run_time(mimic_results)))  
        
        blue_patch = mpatches.Patch(color='blue', label="Simulated Annealing")
        green_patch = mpatches.Patch(color='green', label="Randomized Hill Climbing")
        red_patch = mpatches.Patch(color='red', label="MIMIC")
        purple_patch = mpatches.Patch(color='purple', label="Genetic Algorithm")
        self.clear_plots()                                   
        ax = plt.gca()
        
        self.plot_curves(sa_results, 'blue')
        self.plot_curves(hc_results, 'green')
        self.plot_curves(ga_results, 'purple')
        self.plot_curves(mimic_results, 'red')
        
        plt.legend(handles=[blue_patch, green_patch, red_patch, purple_patch], loc="lower right") 
        plt.xlabel("Iterations")
        plt.ylabel("Fitness Score")
        plt.title("Fitness by Number of Iterations for Each Algorithm")
        N_learning_curve_filename = self.image_path + self.name + "_performance_multiple.png"
        plt.savefig(N_learning_curve_filename) 
        if(self.verbose):
            plt.show()
        
    def plot_curves(self, results, color):
        
        for i in range(len(results)):
            if(len(results[i]) > 3):
                curve = results[i][3]
                plt.plot(curve, color=color)
                
    def get_avg_fitness(self, results):
        avg = 0
        if(len(results) > 0):
            sum = 0
            for r in results:
                sum += r[0]
            avg = sum / len(results)
        return avg
    
    def get_avg_run_time(self, results):
        avg = 0
        if(len(results) > 0):
            sum = 0
            for r in results:
                sum += r[1]
            avg = sum / len(results)
        return avg

    def get_simulated_annealing_run(self, schedule_name, decay, results, random_state=1):
        
        if(schedule_name == "exponential"):
            schedule = mlrose.ExpDecay(exp_const=decay)
        elif(schedule_name == "geometric"):
            schedule = mlrose.GeomDecay(decay=decay)
        elif(schedule_name == "arithmetic"):
            schedule = mlrose.ArithDecay(decay=decay)

    # schedule (schedule object, default: mlrose.GeomDecay()) – Schedule used to determine the value of the temperature parameter.
    # max_attempts (int, default: 10) – Maximum number of attempts to find a better neighbor at each step.
    # max_iters (int, default: np.inf) – Maximum number of iterations of the algorithm.
    # init_state (array, default: None) – 1-D Numpy array containing starting state for algorithm. If None, then a random state is used.

        start_time = time.time()
        state, fitness, curve = mlrose.simulated_annealing(self.problem,
                                                        schedule=schedule,
                                                        max_attempts=self.max_attempts ,
                                                        max_iters=self.max_iterations,
                                                        curve=self.curve,
                                                        init_state=self.init_state,
                                                        random_state=random_state)
        end_time = time.time() - start_time
        
        msg = "Finished Simulated Annealing: " + schedule_name + " " + str(decay) + " in " + str(end_time) + "\n"
        logging.info(msg)
        print(msg)
        
        result = (fitness, end_time, state, curve, (schedule_name, decay))
        results.append(result)
    
    def set_simulated_annealing_threads_fixed_params(self, N, results, schedule_name, decay, random_state=1):
        for i in range(N):
            thread = threading.Thread(target=self.get_simulated_annealing_run, args=(schedule_name, decay, results, i))
            self.threads.append(thread)

    def set_simulated_annealing_threads(self, results):
        
        for schedule in ["exponential", "geometric", "arithmetic"]:
            if(schedule == "geometric"):
                for decay in self.geom_decay:
                    thread = threading.Thread(target=self.get_simulated_annealing_run, args=(schedule, decay, results))
                    self.threads.append(thread)
            if(schedule == "arithmetic"):
                for decay in self.arith_decay:
                    thread = threading.Thread(target=self.get_simulated_annealing_run, args=(schedule, decay, results))
                    self.threads.append(thread)            
            elif(schedule == "exponential"):
                for decay in self.exp_decay:
                    thread = threading.Thread(target=self.get_simulated_annealing_run, args=(schedule, decay, results))
                    self.threads.append(thread)
        
    def simulated_annealing_example(self):
        print("Start Simulated Annealing Example....")  
        
        results = []
        
        self.set_simulated_annealing_threads(results)
        
        self.execute_threads()
        
        print("End Hill Climbing Example.\n\n")     
        
        self.print_fitness_curves(results, "Simulated Annealing", "simulated_annealing_params")
  
        return self.get_best_result(results)
    
    def get_hill_climbing_run(self, restarts, results, random_state=1):
        
    # max_attempts (int, default: 10) – Maximum number of attempts to find a better neighbor at each step.
    # max_iters (int, default: np.inf) – Maximum number of iterations of the algorithm.
    # restarts (int, default: 0) – Number of random restarts.
    # init_state (array, default: None) – 1-D Numpy array containing starting state for algorithm. If None, then a random state is used.

        # prctl.set_name("randomized hill climbing")
        start_time = time.time()
        
        iter = int(self.max_iterations / restarts)
        
        state, fitness, curve = mlrose.random_hill_climb(self.problem,
                                                                   max_attempts=self.max_attempts,
                                                                   max_iters=iter,
                                                                   restarts=restarts,
                                                                   init_state=self.init_state,
                                                                   curve=self.curve,
                                                                   random_state=random_state)
        end_time = time.time() - start_time

        msg = "Finished Hill Climbing: " + str(restarts) + " in " + str(end_time) + "\n"
        logging.info(msg)
        print(msg)

        result = (fitness, end_time, state, curve, restarts)
        results.append(result)

    def set_hill_climbing_threads_fixed_param(self, N, results, random_state=1):
        for i in range(N):
            thread = threading.Thread(target=self.get_hill_climbing_run, args=(self.restarts, results, i))
            self.threads.append(thread)

    def set_hill_climbing_threads(self, results):

        thread = threading.Thread(target=self.get_hill_climbing_run, args=(self.restarts, results))
        self.threads.append(thread)
        
    def hill_climbing_example(self):
        print("Start Randomized Hill Climbing Example....")  

        results = []
        
        self.set_hill_climbing_threads(results)
        
        self.execute_threads()
        
        print("End Hill Climbing Example.\n\n")     
        
        self.print_fitness_curves(results, "Hill Climbing", "hill_climbing_params")
        
        return self.get_best_result(results)

    def get_genetic_run(self, pop, prob, results, random_state=1):
        
        # pop_size (int, default: 200) – Size of population to be used in genetic algorithm.
        # mutation_prob (float, default: 0.1) – Probability of a mutation at each element of the state vector during reproduction, expressed as a value between 0 and 1.
        # max_attempts (int, default: 10) – Maximum number of attempts to find a better state at each step.
        # max_iters (int, default: np.inf) – Maximum number of iterations of the algorithm.

        # prctl.set_name("genetic algorithm")
        try:
            start_time = time.time()
            state, fitness, curve = mlrose.genetic_alg(problem=self.problem,
                                                         pop_size=pop,
                                                         mutation_prob=prob,
                                                        max_attempts=self.max_attempts ,
                                                        max_iters=self.max_iterations,
                                                         curve=self.curve,
                                                         random_state=random_state)
            end_time = time.time() - start_time
    
            msg = "Finished Genetic Alg: " + str(pop) + " " + str(prob) + " in " + str(end_time) + "\n"
            logging.info(msg)
            print(msg)
  
            results.append((fitness, end_time, state, curve, (pop, prob)))
            
        except  Exception as e:
            msg = "Could not compute Genetic Alg: " + str(pop) + " " + str(prob) + "\n"
            logging.info(msg)
            print(msg)            
    
    def set_genetic_threads_fixed_param(self, results, population_size, mutation_prob, random_state=1):
        for i in range(N):
            thread = threading.Thread(target=self.get_genetic_run, args=(population_size, mutation_prob, results, i))
            self.threads.append(thread)
            
    def set_genetic_threads(self, results, population_sizes, mutation_probs):
        for pop in population_sizes:
            for prob in mutation_probs:
                thread = threading.Thread(target=self.get_genetic_run, args=(pop, prob, results))
                self.threads.append(thread)
    
    def genetic_example(self):
        print("Start Genetic Algorithm Example....")  
   
        results = []
        
        self.set_genetic_threads(results, self.population_sizes, self.mutation_probs)
                
        self.execute_threads()
        
        print("End Genetic Algorithm Example.\n\n")     
        
        self.print_fitness_curves(results, "Genetic Algorithm", "genetic_alg_params")
        
        # curve = [ele for ele in curve for i in range(200)]
        
        return self.get_best_result(results)
        
    def get_mimic_run(self, pop, results, random_state=1):
        
    # pop_size (int, default: 200) – Size of population to be used in algorithm.
    # keep_pct (float, default: 0.2) – Proportion of samples to keep at each iteration of the algorithm, expressed as a value between 0 and 1.
    # max_attempts (int, default: 10) – Maximum number of attempts to find a better neighbor at each step.
    # max_iters (int, default: np.inf) – Maximum number of iterations of the algorithm.
    # fast_mimic (bool, default: False) – Activate fast mimic mode to compute the mutual information in vectorized form. Faster speed but requires more memory.

        # prctl.set_name("mimic")
        try:
            
            start_time = time.time()
            state, fitness, curve = mlrose.mimic(problem=self.problem,
                                                   pop_size=pop,
                                                   max_attempts=self.max_attempts ,
                                                   max_iters=self.max_iterations,
                                                   curve=self.curve,
                                                   random_state=random_state)
            end_time = time.time() - start_time
            
            msg = "Finished MIMIC: " + str(pop) + " in " + str(end_time) + "\n"
            logging.info(msg)
            print(msg)
            
            results.append((fitness, end_time, state, curve, (pop)))
        
        except  Exception as e: 
            msg = "Could not compute MIMIC: " + str(pop) + str(e) + "\n"
            logging.info(msg)
            print(msg)      

    def set_mimic_threads_fixed_param(self, N, results, population_size, random_state=1):
        for i in range(N):
            thread = threading.Thread(target=self.get_mimic_run, args=(population_size, results, i))
            self.threads.append(thread)            
        
    def set_mimic_threads(self, results, population_sizes):
        for pop in population_sizes:
            thread = threading.Thread(target=self.get_mimic_run, args=(pop, results))
            self.threads.append(thread)
            # state, fitness, curve = get_mimic(pop)
        
    def mimic_example(self):
        print("Start MIMIC Example....")  

        results = []
        self.set_mimic_threads(results, self.population_sizes)
        self.execute_threads()

        print(results)
      
        print("End MIMIC Example.\n\n")  
        
        self.print_fitness_curves(results, "MIMIC", "mimic_params")
        
        # curve = [ele for ele in curve for i in range(200)]
        
        return self.get_best_result(results)

    def get_best_result(self, results):
        
        best_fitness = None
        best_result = None

        for result in results:
            fitness = None
            if(len(result) > 0):
                fitness = result[0]

            if(best_fitness == None or (fitness != None and fitness > best_fitness)):
                best_fitness = fitness
                best_result = result
        
        return best_result
    
    def get_avg_run_time(self, results):
        
        sum = 0.0

        try:
            for result in results:
                sum += result[1]
        except:
            pass
        
        avg = 0.0
        if(len(results) > 0):
            avg = sum / len(results)
            
        return avg
    
    def get_best_curve(self, result):
        try:
            return result[3]
        except:
            return []
    
    def print_fitness_curves(self, results, algorithm_title, file_label):
        
        self.clear_plots()                                   
        ax = plt.gca()
        
        L = len(results)
        
        for i in range(L):
            try:
                curve = results[i][3]
                plt.plot(curve)
            except:
                pass

        plt.title("Fitness For " + algorithm_title + " Over Parameter Choices")
        plt.xlabel("Iterations")
        plt.ylabel("Fitness Score")

        filename = self.image_path + self.name + "_" + file_label + ".png"     
        plt.savefig(filename)
        if(self.verbose):
            plt.show()
              
