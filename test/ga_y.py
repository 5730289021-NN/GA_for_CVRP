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
