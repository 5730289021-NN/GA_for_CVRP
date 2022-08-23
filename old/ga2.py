import numpy as np
import pandas as pd
import math


# Find euclidean distance of a and b
# a, b must be a tuple of location ie. x, y

def euclidean_dist(a, b):
    return round(math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))
    #return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) # for better precision

depot_param = np.array([[99, 1, -1, 0]]) # index, x, y, demand

# Read from csv
#city_position_demand = np.concatenate((depot_param, pd.read_csv('A-n32-k5.csv').to_numpy()))

# Read from excel 
# [Required] pip3 install openpyxl
city_position_demand = np.concatenate((pd.read_excel('A-n32-k5.xlsx').to_numpy(), depot_param))
# print(city_position_demand)

position = city_position_demand[:, 1:3]
demand = city_position_demand[:, 3]

n_node = len(city_position_demand)
print("Total Nodes:", n_node)


dist_table = np.zeros((n_node, n_node))

# print(position)
# print(demand)


for i in range(n_node):
    for j in range(n_node):
        if i == j:
            dist_table[i,j] = 999
        else:
            dist_table[i,j] = euclidean_dist(position[i], position[j])

np.savetxt('t1.csv', dist_table, delimiter=",", fmt='%d')


city_demand = np.concatenate((city_position_demand[:,0].reshape(1, len(city_position_demand[:,0])), 
                              city_position_demand[:,3].reshape(1, len(city_position_demand[:,0]))), axis=0)
print(city_demand.transpose())