import numpy as np
import disropt as dp
import pygad
from typing import Union, Callable
from threading import Event
from disropt.agents.agent import Agent
from disropt.problems.problem import Problem
from disropt.problems.constraint_coupled_problem import ConstraintCoupledProblem
from disropt.functions import Variable
from disropt.functions import ExtendedFunction
from disropt.constraints import ExtendedConstraint
from algorithm import Algorithm
        
class GAProblem(Problem):            
    def ga_run(self, x_shape, cur_x = None, smart_init = True, ga_seed = 1):
        self.x_shape = x_shape
        desired_output = -1000 #Set it as very small if it is to minimise it
        
        ## GA Parameters
        #Load in information about the genetic algorithm
        num_generations = 50
        num_parents_mating = 30
        sol_per_pop = 50
        
        num_genes = self.x_shape[0] + 1 + len(self.constraints)
        parent_selection_type = "sss"
        keep_parents = 30
        crossover_type = "single_point"
        mutation_type = "random"
        mutation_percent_genes = 20
        
        init_range_low = -2.0
        init_range_high = 2.0
        gene_space = None
        if smart_init:
            if not cur_x is None:
                x_gene_space = {'low': -(cur_x[0][0]), 'high': (cur_x[0][0])}
                gene_space = []
                for i in range(self.x_shape[0] + 1):
                    gene_space.append(x_gene_space)
                for i in range(len(self.constraints)):
                    gene_space.append({'low':-10, 'high':10})
        #Create the genetic algorithm instance, with its objective function
        fitness_function = self.ga_fitness_function
        if gene_space != None:
            ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           init_range_low = init_range_low,
                           init_range_high = init_range_high,
                           gene_space=gene_space,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           random_seed = ga_seed)
        else:
            ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           init_range_low = init_range_low,
                           init_range_high = init_range_high,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           random_seed = ga_seed)
        #Run the GA
        ga_instance.run()
        #Retrieve the best results from the GA
        ga_results = ga_instance.best_solution()
        x_values = []
        for i in range(self.x_shape[0] + 1):
            x_values.append([ga_results[0][i]])
        x_values = np.asarray(x_values)
        i += 1
        lambdas = []
        for j in range(len(self.constraints)):
            lambdas.append(ga_results[0][i + j])
        return {'solution':x_values, 'dual_variables':np.asarray(lambdas)}
        
    def ga_fitness_function(self, inputs, desired_solution):
        x_values = []
        for i in range(self.x_shape[0] + 1):
            x_values.append([inputs[i]])
        x_values = np.asarray(x_values)
        lambdas = []
        for j in range(len(self.constraints)):
            lambdas.append(inputs[i + j])
        results = self.lagrangian(x_values, lambdas, self.constraints)
        return results[0][0]
        
    def lagrangian(self, x_mtx, lambdas, constraints):
        constraint_sum = 0
        for i in range(len(lambdas)): #For each lambda (or constraint)
            #Add the sum of all components produced from the evaluation of the constraints function, multiplied by the respective lambda
            constraint_sum += (np.sum(constraints[i].function.eval(x_mtx)) * lambdas[i])
        return self.objective_function.eval(x_mtx) + constraint_sum