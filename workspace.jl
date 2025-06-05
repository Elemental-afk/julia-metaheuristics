#This file is a dirty "backend" where I tested all of these functions
#I suggest checking out the notebook.jl file for a more organized view of the code


using Random
using LinearAlgebra

function random_array(n::Int)
    list = []
    for i in 1:n
        push!(list, trunc(Int, rand()*1000))
    end
    return Int.(list)
end

println("Random array of 10 elements:")
println(random_array(10))
array = random_array(10)
println(sort(array))


function is_sorted(array::Vector{Int})
    n = length(array)
    for i in 1:n-1
        if array[i+1] < array[i]
            return false
        end
    end
    return true
end

function cost(array::Vector{Int})
    cost = 0
    n = length(array)
    for i in 1:n-1
        for j in i+1:n
            if array[i] > array[j]
                cost += 1
            end
        end
    end
    return cost
end

function swap_array(array::Vector{Int}, mode::String="full_rand")
    new_arr = copy(array)
    n = length(array)
    if mode == "full_rand"
        indices = randperm(n)
        for i in 1:n
            new_arr[i], new_arr[indices[i]] = new_arr[indices[i]], new_arr[i]
        end
    elseif mode == "neigh"
        swap = rand(1:n-1)
        new_arr[swap], new_arr[swap+1] = new_arr[swap+1], new_arr[swap]
    elseif mode == "2rand"
        indices = rand(1:n, 2)
        new_arr[indices[1]], new_arr[indices[2]] = new_arr[indices[2]], new_arr[indices[1]]
    elseif mode == "none"
        1 == 1
    end
    
    new_cost = cost(new_arr)

    
    return new_arr, new_cost
end

function annealing(temperature::Float64, array::Vector{Int}, mode::String="full_rand", alpha::Float64=0.99, max_iter::Int=1000)
    current_array = copy(array)
    current_cost = cost(current_array)
    best_array = copy(current_array)
    best_cost = current_cost
    
    for iter in 1:max_iter
        new_array, new_cost = swap_array(current_array, mode)
        temperature *= alpha 

        if new_cost < current_cost || rand() < exp((current_cost - new_cost)/temperature)
            current_array = new_array
            current_cost = new_cost
            
            if new_cost < best_cost
                best_array = copy(new_array)
                best_cost = new_cost
            end
        end
        
        if best_cost == 0
            break
        end
        
        if iter % 1000 == 0
            println("Iteration $iter: Cost = $current_cost")
        end
    end
    
    return best_array, best_cost
end



# Test code 

array = random_array(100)
println("Starting annealing with initial array:")
println(array)
result_array, final_cost = annealing(10000.0, array, "2rand", 0.999, 100000)
println("Final array after annealing:")
println(result_array)
println("Final cost (number of inversions): ", final_cost)

#Genetic Algorithm solution

function generate_population(pop_size::Int, starting_array::Vector{Int})::Vector{Tuple{Vector{Int}, Int}}
    population = Vector{Tuple{Vector{Int}, Int}}()
    for _ in 1:pop_size
        new_pop = swap_array(starting_array, "full_rand")[1]
        fitness = cost(new_pop)
        push!(population, (new_pop, fitness))
    end
    return population
end

function select_parents(population::Vector{Tuple{Vector{Int}, Int}}, num_parents::Int)
    population_sample = rand(population, num_parents)
    population_sample = sort(population_sample, by=x -> x[2])
    return population_sample
end

function marry_and_reproduce(parent1::Vector{Int}, parent2::Vector{Int}, mutation_rate::Float64=0.01)
    n = length(parent1)
    child = fill(-1, n)

    start_index = rand(1:n)
    end_index = rand(start_index:n)

    child[start_index:end_index] = parent1[start_index:end_index]

    inherited = Set(child[start_index:end_index])

    insert_pos = 1
    for gene in parent2
        if !(gene in inherited)
            while insert_pos <= n && child[insert_pos] != -1
                insert_pos += 1
            end
            if insert_pos <= n
                child[insert_pos] = gene
            end
        end
    end


    if rand() < mutation_rate
        child = swap_array(child, "2rand")[1]
    end

    return child
end

function genetic_algorithm(starting_array::Vector{Int}, pop_size::Int, num_generations::Int, mutation_rate::Float64=0.01)
    population = generate_population(pop_size, starting_array)

    for i in 1:num_generations
        parents = select_parents(population, pop_size ÷ 2)
        new_population = []

        for j in 1:2:length(parents)
            parent1 = parents[j][1]
            parent2 = parents[j+1][1]
            child = marry_and_reproduce(parent1, parent2, mutation_rate)
            fitness = cost(child)
            push!(new_population, (child, fitness))
        end

        population = vcat(population, new_population)
        population = sort(population, by=x -> x[2])
        population = population[1:pop_size]

        population = [(x[1], x[2]) for x in population]
        if population[1][2] == 0
            break
        end
        if i % 100 == 0
            println("Generation: ", i)
            println("Best cost: ", population[1][2])
        end

    end

    best_solution = population[1][1]
    return best_solution, population[1][2]
    
end

array = random_array(50)
println("Starting GA with initial array:")
println(array)
result_array, final_cost = genetic_algorithm(array, 100, 2000, 0.9)
println("Final array after GA:")
println(result_array)
println("is perm of initial array: ", sort(result_array) == sort(array))

#Ant Colony Optimization

function select_swap(current, pheromones, alpha, beta, heuristic)
    n = length(current)
    possible_swaps = Tuple{Int,Int}[]
    probabilities = Float64[]
    
    for i in 1:n-1
        for j in i+1:n
            if current[i] != current[j]
                tau = pheromones[i,j] ^ alpha
                eta = heuristic(i, j, current) ^ beta
                prob = tau * eta
                push!(possible_swaps, (i, j))
                push!(probabilities, prob)
            end
        end
    end
    
    if isempty(possible_swaps)
        return rand(1:n, 2)
    end
    
    prob_sum = sum(probabilities)
    if prob_sum == 0
        return rand(possible_swaps)
    end
    probabilities ./= prob_sum
    
    r = rand()
    cumulative_prob = 0.0
    for (i, swap) in enumerate(possible_swaps)
        cumulative_prob += probabilities[i]
        if r <= cumulative_prob
            return swap
        end
    end
    
    return possible_swaps[end]
end

function ant_colony_sort(arr, num_ants=30, max_iter=100, evaporation_rate=0.2, alpha=2, beta=50)
    n = length(arr)
    if n <= 1
        return copy(arr)
    end

    function heuristic(i, j, current)
        temp = copy(current)
        temp[i], temp[j] = temp[j], temp[i]
        return 1.0 / (1 + cost(temp))
    end

    pheromones = fill(1.0, n, n)
    best_order = copy(arr)
    best_inversions = cost(arr)

    for iter in 1:max_iter
        ant_solutions = Vector{Tuple{Vector{Int}, Int}}()
        
        for _ in 1:num_ants
            current = copy(arr)
            for _ in 1:n
                i, j = select_swap(current, pheromones, alpha, beta, heuristic)
                current[i], current[j] = current[j], current[i]
            end
            inv = cost(current)
            push!(ant_solutions, (current, inv))

            if inv < best_inversions
                best_order = copy(current)
                best_inversions = inv
                if best_inversions == 0
                    println("Sorted in iteration $iter")
                    return best_order
                end
            end
        end

        pheromones .*= (1 - evaporation_rate)

        sorted_ants = sort(ant_solutions, by = x -> x[2])
        top_k = max(1, Int(round(num_ants * 0.1)))

        for (ant_arr, inv) in sorted_ants[1:top_k]
            for i in 1:n-1
                for j in i+1:n
                    if ant_arr[i] > ant_arr[j]
                        pheromones[i,j] += 2.0 / (1 + inv)
                    end
                end
            end
        end

        if iter % 2 == 0
            println("Iter $iter — Best inversions: $best_inversions")
        end
    end

    return best_order
end

arr = shuffle(1:50)
sorted_arr = ant_colony_sort(arr)
println("Sorted? ", is_sorted(sorted_arr))

