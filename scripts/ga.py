import networkx as nx
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import pygad
import random

# Find euclidean distance of a and b
# a, b must be a tuple of location ie. x, y

def euclidean_dist(a, b):
    return round(math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))
    #return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) # for better precision

depot_param = np.array([[0, 1, -1, 0]]) # index: 0 depot, x, y, demand

TRUCK_CAPACITY = 10

# Read from excel 
city_position_demand = np.concatenate((depot_param,pd.read_excel('data/A-n32-k5.xlsx').to_numpy()))
# print(city_position_demand)
# [ 0  1 -1  0]
# [ 1 82 76  0]
# [ 2 96 44 19]
# ...

G = nx.Graph()

for r in city_position_demand:
    G.add_node(r[0], pos=(r[1],r[2]), demand=r[3])

position = city_position_demand[:, 1:3]
node_amount = len(city_position_demand)
distance_table = np.zeros((node_amount, node_amount))
for i in range(node_amount):
    for j in range(node_amount):
        if i == j:
            distance_table[i,j] = 0 # same location has zero cost
        else:
            distance_table[i,j] = euclidean_dist(position[i], position[j])
            # if i == 0:
            #     G.add_edge("Source", j, cost=euclidean_dist(position[i], position[j]))    
            # elif j == 0:
            #     G.add_edge(i, "Sink", cost=euclidean_dist(position[i], position[j])) 
            # else:
            #     G.add_edge(i, j, cost=euclidean_dist(position[i], position[j]))
            G.add_edge(i, j, cost=euclidean_dist(position[i], position[j]))
            
pos_dict = nx.get_node_attributes(G,'pos')
demand_dict = nx.get_node_attributes(G, 'demand')
cost_dict = nx.get_edge_attributes(G, 'cost')

print(cost_dict)
nx.draw_networkx(G, with_labels = True, pos=pos_dict)
# nx.draw_networkx_nodes(G, pos)
# nx.draw_networkx_labels(G, pos)
# nx.draw_networkx_edges(G, pos, edge_color='r', arrows = True)

# plt.grid()
# plt.show()
# city_demand = np.concatenate((city_position_demand[:,0].reshape(1, len(city_position_demand[:,0])), 
#                               city_position_demand[:,3].reshape(1, len(city_position_demand[:,0]))), axis=0)
# [ 0  0]
# [ 1  0]
# [ 2 19]
# [ 3 21]
# print(city_demand.transpose())


def fitness_func(solution, solution_idx):
    # Route starts from depot and sink at depot always
    # Solution format: [[0, 1, 2, 0], [0, 5, 7, 0], [...], [...]] for 4 routes
    # 1. [Filter out bad solution] Check if the solution is valid else return inf
    # 1.1 Check if all routes passes all the nodes and must pass each node only once
    # 1.2 Check if all demand in routes must not beyond truck capacity
    passing_nodes = np.ones(node_amount) # including depot
    passing_nodes[0] = len(solution) # Trucks must passed depot for 2 * n_route times
    for route in solution:
        truck_capacity_r = TRUCK_CAPACITY
        for node in route:#  at depot demand = 0 
            truck_capacity_r -= G[node]['demand']
            passing_nodes[node] = passing_nodes[node] - 1
        if truck_capacity_r < 0:
            return -math.inf
    # At this point `passing_nodes` must be np.zeros(node_amount) else bad answer
    if np.any(passing_nodes): # Has not zero at some points
        return -math.inf
    # 2. Calculate distances
    cost = 0
    for route in solution:
        for idx, x in enumerate(route):
            if idx > 0 and x == 0:
                pass
            else:
                cost+= cost_dict[(x, route[idx + 1])]

    fitness = 1.0 / cost
    return fitness


# parents: The selected parents.
# offspring_size: 
#   The size of the offspring as a tuple of 2 numbers: 
#       the offspring size: amount of offspring...easy [used]
#       number of genes: can be vary... 4 routes vs 5 routes [unused]
# ga_instance: The instance from the pygad.GA class. This instance helps to retrieve any property like population, gene_type, gene_space, etc.
def crossover_func(parents, offspring_size, ga_instance):
    if len(parents) != 2:
        print('Parent should be 2')
    # Crossover by swapping at route level
    # Ex.   Parent A has 2 routes aka [A B]
    #       Parent B has 3 routes [D E F]
    # Offspring will have both 2 routes and 3 routes
    # Ex. 
    #   2 routes [A D], [A E], [A F] [B D], [B E], [B F]
    #   3 routes [A E F], [D A F], [D E A], [B E F], [D B F], [D E B]
    route_size = [len(parents[0]), len(parents[1])]

    offsprings = []
    for route_i, i in enumerate(parents[0]):
        for route_j in parents[1]:
            p0 = parents[0]
            p0[i] = route_j
            offsprings.append(p0)
    
    for route_i, i in enumerate(parents[1]):
        for route_j in parents[0]:
            p1 = parents[1]
            p1[i] = route_j
            offsprings.append(p1)

    # remove random offspring depend on user input
    survive_offsprings = offsprings
    if offspring_size[0] > len(offsprings):
        print(f'offspring size parameter is too much, max at {len(offsprings)}')
    elif offspring_size[0] < len(offsprings):
        # remove some offsprings
        amount_to_remove = len(offsprings) - offspring_size[0]
        offsprint_to_remove = random.sample(range(len(offsprings)), amount_to_remove)
        survive_offsprings = [os for os, i in enumerate(offsprings) if i not in offsprint_to_remove]

    for offspring in survive_offsprings: # for loop is pass by reference
        fix_chromosome(offspring) # not implemented yet
        
    return np.array(survive_offsprings)



fitness_function = fitness_func

num_generations = 50
num_parents_mating = 4

sol_per_pop = 8
num_genes = len(function_inputs)

init_range_low = -2
init_range_high = 5

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10


ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)