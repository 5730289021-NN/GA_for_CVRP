import random
from random import randrange
from time import time

#program 196
class Problem_Genetic(object):
    def __init__(self,genes,individuals_length,decode,fitness):
        self.genes= genes
        self.individuals_length= individuals_length
        self.decode= decode
        self.fitness= fitness

    def mutation(self, chromosome, prob):
    
        def inversion_mutation(chromosome_aux):
            chromosome = chromosome_aux
            index1 = randrange(0,len(chromosome))
            index2 = randrange(index1,len(chromosome))
            chromosome_mid = chromosome[index1:index2]
            chromosome_mid.reverse()
            chromosome_result = chromosome[0:index1] + chromosome_mid + chromosome[index2:]
            
            return chromosome_result
 
        aux = []
        for _ in range(len(chromosome)):
            if random.random() < prob :
                aux = inversion_mutation(chromosome)
        return aux

    def crossover(self,parent1, parent2):
        def process_gen_repeated(copy_child1,copy_child2):
            count1=0
            for gen1 in copy_child1[:pos]:
                repeat = 0
                repeat = copy_child1.count(gen1)
                if repeat > 1:
                    count2=0
                    for gen2 in parent1[pos:]:
                        if gen2 not in copy_child1:
                            child1[count1] = parent1[pos:][count2]
                        count2+=1
                count1+=1
            count1=0
            for gen1 in copy_child2[:pos]:
                repeat = 0
                repeat = copy_child2.count(gen1)
                if repeat > 1:
                    count2=0
                    for gen2 in parent2[pos:]:
                        if gen2 not in copy_child2:
                            child2[count1] = parent2[pos:][count2]
                        count2+=1
                count1+=1
            return [child1,child2]

        pos=random.randrange(1,self.individuals_length-1)
        child1 = parent1[:pos] + parent2[pos:] 
        child2 = parent2[:pos] + parent1[pos:]

        return process_gen_repeated(child1, child2)
 
 
def decodeVRP(chromosome): 
    list=[]
    for (k,v) in chromosome:
        if k in trucks[:(num_trucks-1)]:
            list.append(frontier)
            continue
        list.append(cities.get(k))
    return list

def penalty_capacity(chromosome):
    actual = chromosome
    value_penalty = 0
    capacity_list = []
    index_cap = 0
    overloads = 0

    for i in range(0,len(trucks)):
        init = 0
        capacity_list.append(init)

    for (k,v) in actual:
        if k not in trucks:
            capacity_list[int(index_cap)]+=v
        else:
            index_cap+= 1

        if capacity_list[index_cap] > capacity_trucks:
            overloads+=1
            value_penalty+= 100 * overloads
    return value_penalty

def fitnessVRP(chromosome):

    def distanceTrip(index,city):
        w = distances.get(index)
        return w[city]

    actualChromosome = chromosome
    fitness_value = 0

    penalty_cap = penalty_capacity(actualChromosome)
    for (key,value) in actualChromosome:
        if key not in trucks:
            nextCity_tuple = actualChromosome[key]
            if list(nextCity_tuple)[0] not in trucks:
                nextCity= list(nextCity_tuple)[0]
                fitness_value+= distanceTrip(key,nextCity) + (50 * penalty_cap)
    return fitness_value

#program 197
def genetic_algorithm_t(Problem_Genetic,k,opt,ngen,size,ratio_cross,prob_mutate):

    def initial_population(Problem_Genetic,size): 
        def generate_chromosome():
            chromosome=[]
            for i in Problem_Genetic.genes:
                chromosome.append(i)
            random.shuffle(chromosome)
            return chromosome
        return [generate_chromosome() for _ in range(size)]

    def new_generation_t(Problem_Genetic,k,opt,population,n_parents,n_directs,prob_mutate):
        def tournament_selection(Problem_Genetic,population,n,k,opt):
            winners=[]
            for _ in range(n):
                elements = random.sample(population,k)
                winners.append(opt(elements,key=Problem_Genetic.fitness))
            return winners

        def cross_parents(Problem_Genetic,parents):
            childs=[]
            for i in range(0,len(parents),2):
                childs.extend(Problem_Genetic.crossover(parents[i],parents[i+1]))
            return childs
    
        def mutate(Problem_Genetic,population,prob):
            for i in population:
                Problem_Genetic.mutation(i,prob)
            return population

        directs = tournament_selection(Problem_Genetic, population, n_directs, k, opt)
        crosses = cross_parents(Problem_Genetic,
        tournament_selection(Problem_Genetic, population, n_parents, k, opt))
        mutations = mutate(Problem_Genetic, crosses, prob_mutate)
        new_generation = directs + mutations

        return new_generation

    population = initial_population(Problem_Genetic, size)
    n_parents = round(size*ratio_cross)
    n_parents = (n_parents if n_parents%2==0 else n_parents-1)
    n_directs = size - n_parents

    for _ in range(ngen):
        population = new_generation_t(Problem_Genetic, k, opt, population, n_parents, n_directs, prob_mutate)

    bestChromosome = opt(population, key = Problem_Genetic.fitness)
    print("Chromosome: ", bestChromosome)
    genotype = Problem_Genetic.decode(bestChromosome)
    print("Solution: " , (genotype,Problem_Genetic.fitness(bestChromosome)))
    return (genotype,Problem_Genetic.fitness(bestChromosome))

#program 198
def genetic_algorithm_t2(Problem_Genetic,k,opt,ngen,size,ratio_cross,prob_mutate,dictionary):
    def initial_population(Problem_Genetic,size): 
        def generate_chromosome():
            chromosome=[]
            for i in Problem_Genetic.genes:
                chromosome.append(i)
            random.shuffle(chromosome)
            dictionary[str(chromosome)]=1
            return chromosome
        return [generate_chromosome() for _ in range(size)]

    def new_generation_t(Problem_Genetic,k,opt,population,n_parents,n_directs,prob_mutate):
        def tournament_selection(Problem_Genetic,population,n,k,opt):
            winners = []
            for _ in range(int(n)):
                elements = random.sample(population,k)
                winners.append(opt(elements,key=Problem_Genetic.fitness))
            for winner in winners:
                if str(winner) in dictionary:
                    dictionary[str(winner)]=dictionary[str(winner)]+1
                else:
                    dictionary[str(winner)]=1
                return winners

        def cross_parents(Problem_Genetic, parents):
            childs=[]
            for i in range(0,len(parents),2):
                childs.extend(Problem_Genetic.crossover(parents[i],parents[i+1]))
                parent = str(parents[i])
                if parent not in dictionary:
                    dictionary[parent]=1
                dictionary[str(childs[i])] = dictionary[parent]
                del dictionary[str(parents[i])]
            return childs

        def mutate(Problem_Genetic,population,prob):
            j = 0
            copy_population=population
            for crom in population:
                Problem_Genetic.mutation(crom,prob)
                parent = str(crom) 
                if parent in dictionary:
                    dictionary[str(population[j])] = dictionary[parent] 
                    del dictionary[str(copy_population[j])]
                    j+=j
            return population

        directs = tournament_selection(Problem_Genetic, population, n_directs, k, opt)
        crosses = cross_parents(Problem_Genetic,tournament_selection(Problem_Genetic, population, n_parents, k, opt))
        mutations = mutate(Problem_Genetic, crosses, prob_mutate)
        new_generation = directs + mutations
        for ind in new_generation:
            age = 0
            crom = str(ind)
            if crom in dictionary:
                age+= 1
                dictionary[crom]+= 1
            else:
                dictionary[crom] = 1
        return new_generation

    population = initial_population(Problem_Genetic, size )
    n_parents= round(size*ratio_cross)
    n_parents = (n_parents if n_parents%2==0 else n_parents-1)
    n_directs = size - n_parents

    for _ in range(ngen):
        population = new_generation_t(Problem_Genetic, k, opt, population, n_parents, n_directs, prob_mutate)

    bestChromosome = opt(population, key = Problem_Genetic.fitness)
    print("Chromosome: ", bestChromosome)
    genotype = Problem_Genetic.decode(bestChromosome)
    print("Solution: ", (genotype,Problem_Genetic.fitness(bestChromosome)), dictionary[(str(bestChromosome))], " GENERATIONS.")

    return (genotype,Problem_Genetic.fitness(bestChromosome) + dictionary[(str(bestChromosome))]*50)

def VRP(k):
    VRP_PROBLEM = Problem_Genetic([(0,10),(1,10),(2,10),(3,10),(4,10),(5,10),(6,10),(7,10),(trucks[0],capacity_trucks)], len(cities), lambda x : decodeVRP(x), lambda y: fitnessVRP(y))
    
    def first_part_GA(k):
        cont = 0
        print ("---------------------------------------------------------Executing FIRST PART: VRP --------------------------------------------------------- \n")
        print("Capacity of trucks = ", capacity_trucks)
        print("Frontier = ", frontier)
        print("")
        tiempo_inicial_t2 = time()
        while cont <= k: 
            genetic_algorithm_t(VRP_PROBLEM, 2, min, 200, 100, 0.8, 0.05)
            cont+=1
        tiempo_final_t2 = time()
        print("\n") 
        print("Total time: ",(tiempo_final_t2 - tiempo_inicial_t2)," secs.\n")

    def second_part_GA(k):
        print ("---------------------------------------------------------Executing SECOND PART: VRP --------------------------------------------------------- \n")
        print("Capacity of trucks = ", capacity_trucks)
        print("Frontier = ", frontier)
        print("")

        cont = 0
        tiempo_inicial_t2 = time()
        while cont <= k: 
            genetic_algorithm_t2(VRP_PROBLEM, 2, min, 200, 100, 0.8, 0.05,{})
            cont+=1
        tiempo_final_t2 = time()
        print("|n") 
        print("Total time: ",(tiempo_final_t2 - tiempo_inicial_t2)," secs.\n")
        
    first_part_GA(k)
    print("-------------------------------------------------------------------------------------------------------------------------------------------------------")
    second_part_GA(k)

cities = {0:'Almeria', 1:'Cadiz', 2:'Cordoba', 3:'Granada', 4:'Huelva', 5:'Jaen', 6:'Malaga', 7:'Sevilla'}
#Distance between each pair of cities
w0 = [999,454,317,165,528,222,223,410]
w1 = [453,999,253,291,210,325,234,121]
w2 = [317,252,999,202,226,108,158,140]
w3 = [165,292,201,999,344,94,124,248]
w4 = [508,210,235,346,999,336,303,94]
w5 = [222,325,116,93,340,999,182,247]
w6 = [223,235,158,125,302,185,999,206]
w7 = [410,121,141,248,93,242,199,999]
distances = {0:w0, 1:w1, 2:w2, 3:w3, 4:w4, 5:w5, 6:w6, 7:w7}
capacity_trucks = 60
trucks = ['truck', 'truck']
num_trucks = len(trucks)
frontier = "---------"

if __name__ == "__main__":
    genetic_problem_instances = 10
    print("EXECUTING ", genetic_problem_instances, " INSTANCES ")
    VRP(genetic_problem_instances)
