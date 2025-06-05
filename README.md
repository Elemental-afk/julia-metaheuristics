
# Metaheuristic Sorting Algorithms in Julia

*A computational exploration of optimization techniques for sorting problems*

## Project Overview

This Pluto.jl notebook implements and compares three metaheuristic approaches to array sorting:

1. **Simulated Annealing** - Inspired by metallurgical cooling processes
2. **Genetic Algorithms** - Mimicking biological evolution
3. **Ant Colony Optimization** - Modeling ant foraging behavior

While these methods are typically used for NP-hard problems, this project adapts them to sorting - a solved problem - as an educational exercise in algorithm design and Julia programming.

## Key Features

- Interactive visualizations of sorting processes
- Comparative performance analysis
- Customizable parameters for each algorithm
- Cost function implementations for inversion counting

## Implementation Details

### Core Algorithms

```julia
# Simulated Annealing
function annealing(temperature, array, mode, alpha, max_iter)
    # Cooling schedule and probabilistic acceptance
    ...
end

# Genetic Algorithm
function genetic_algorithm(array, pop_size, generations, mutation_rate)
    # Selection, crossover and mutation
    ...
end

# Ant colony optimization
function ant_colony_sort(arr, num_ants=30, max_iter=100, evaporation_rate=0.2, alpha=2, beta=50)
    # Pheromone and probability logic
    ...
end
```
## Results

Preliminary benchmarks on arrays of size n=50:

| Method               | Time (ms) | Relative Performance |
|----------------------|-----------|----------------------|
| Native `sort()`      | 0.05      | 1.0x (baseline)      |
| Simulated Annealing  | 420       | 8400x                |
| Genetic Algorithm    | 690       | 13800x               |
| Ant Colony           | too long to bother      | just don't              |

## How to Use

1. Install Julia 1.6+ and Pluto.jl
2. Open the notebook in Pluto
3. Modify parameters using the interactive controls
4. Run cells to observe algorithm behavior
