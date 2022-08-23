from networkx import DiGraph
import numpy as np
import pandas as pd
import math

# Find euclidean distance of a and b
# a, b must be a tuple of location ie. x, y

def euclidean_dist(a, b):
    return round(math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))
    #return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) # for better precision

depot_param = np.array([[0, 1, -1, 0]]) # index: 0 depot, x, y, demand

# Read from excel 
city_position_demand = np.concatenate((depot_param,pd.read_excel('data/A-n32-k5.xlsx').to_numpy()))
# print(city_position_demand)
# [ 0  1 -1  0]
# [ 1 82 76  0]
# [ 2 96 44 19]
# ...


position = city_position_demand[:, 1:3]
# print(position)
# [[ 1 -1]
#  [82 76]
#  [96 44]
#  ...

node_amount = len(city_position_demand)
print("Total Nodes:", node_amount)

distance_table = np.zeros((node_amount, node_amount))

for i in range(node_amount):
    for j in range(node_amount):
        if i == j:
            distance_table[i,j] = 0 # same location has zero cost
        else:
            distance_table[i,j] = euclidean_dist(position[i], position[j])

# np.savetxt('t1.csv', distance_table, delimiter=",", fmt='%d')
dt_row, dt_column = distance_table.shape
distance_matrix = np.pad(distance_table, ((0,1),(0,1)), 'constant') #((top,bottom),(left,right))
# print(distance_matrix)
distance_matrix[:, -1] = distance_matrix[:,0]
# print(distance_matrix)
distance_matrix[:,0] = 0
# print(distance_matrix)




city_demand = np.concatenate((city_position_demand[:,0].reshape(1, len(city_position_demand[:,0])), 
                              city_position_demand[:,3].reshape(1, len(city_position_demand[:,0]))), axis=0)
# [ 0  0]
# [ 1  0]
# [ 2 19]
# [ 3 21]
# print(city_demand.transpose())