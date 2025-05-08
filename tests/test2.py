from railway import *

# Load model

# Define problem parameters
P = 2000
K = 3
timelimit = 60
SAtimelimit = 30

# N = 10; J = 10; T = 10  #    1
# N = 10; J = 10; T = 50  #    2
# N = 10; J = 10; T = 100 #    3
# N = 10; J = 40; T = 10  #    4
# N = 10; J = 40; T = 50  #    5
# N = 10; J = 40; T = 100 #    6
# N = 10; J = 80; T = 10  #    7
# N = 10; J = 80; T = 50  #    8
N = 10; J = 80; T = 100 #    9
# N = 20; J = 10; T = 10  #   10
# N = 20; J = 10; T = 50  #   11
# N = 20; J = 10; T = 100 #   12
# N = 20; J = 40; T = 10  #   13
# N = 20; J = 40; T = 50  #   14
# N = 20; J = 40; T = 100 #   15
# N = 20; J = 80; T = 10  #   16
# N = 20; J = 80; T = 50  #   17
# N = 20; J = 80; T = 100 #   18
# N = 40; J = 10; T = 10  #   19
# N = 40; J = 10; T = 50  #   20
# N = 40; J = 10; T = 100 #   21
# N = 40; J = 40; T = 10  #   22
# N = 40; J = 40; T = 50  #   23
# N = 40; J = 40; T = 100 #   24
# N = 40; J = 80; T = 10  #   25
# N = 40; J = 80; T = 50  #   26
# N = 40; J = 80; T = 100 #   27

# Name of the file to load
FILENAME = f"datasets/railway_N{N}_T{T}_J{J}_P{P}_K{K}.json"



# Model 0: "as-is" Gurobi model with no heuristics or cuts
print("\nModel 0\n")
model0 = Railway.load(FILENAME)
# model0.model.setParam('OutputFlag', 0) # verbose
model0.model.setParam('TimeLimit', timelimit) # time limit
model0.model.setParam('LPWarmStart', 0)
model0.model.setParam('PoolSolutions', 1)
model0.model.setParam('Cuts', 0)
model0.model.setParam('CutPasses', 0)
model0.model.setParam('Heuristics', 0)
model0.model.setParam('Symmetry', 0)
model0.model.setParam('Threads', 1)
model0.model.setParam('Presolve', 0)

model0.set_constraints()
model0.set_objective()

results0 = model0.optimize()

print('-----------------------------------')
print(f'Status: {model0.get_status()}')
print("Runtime:", results0['runtime'])
print("Gap:", results0['gap'])
print("Objective value:", results0['obj'])



# Model 1: model with simulated annealing heuristic
print("\nModel 1\n")
model1 = Railway.load(FILENAME)
# model1.model.setParam('OutputFlag', 0) # verbose
model1.model.setParam('TimeLimit', timelimit) # time limit
# model1.model.setParam('LPWarmStart', 1) # Must use warm start from SA
model1.model.setParam('PoolSolutions', 1)
model1.model.setParam('Cuts', 0)
model1.model.setParam('CutPasses', 0)
model1.model.setParam('Heuristics', 0)
model1.model.setParam('Symmetry', 0)
model1.model.setParam('Threads', 1)
model1.model.setParam('Presolve', 0)

print('Running simulated annealing...')
S, SAtime = model1.simulated_annealing(
    T=5e5,
    c=0.99,
    L=1,
    min_T=1,
    max_time=SAtimelimit
)
model1.set_solution(S)

model1.set_constraints()
model1.set_objective()

results1 = model1.optimize()

print('-----------------------------------')
print(f'Status: {model1.get_status()}')
print("Simulated Annealing time:", SAtime)
print("Runtime:", results1['runtime'])
print("Total time:", SAtime + results1['runtime'])
print("Gap:", results1['gap'])
print("Objective value:", results1['obj'])



# Model 2: Full model plus valid inequalities and cutting planes
print("\nModel 2\n")
model2 = Railway.load(FILENAME)
# model2.model.setParam('OutputFlag', 0) # verbose
model2.model.setParam('TimeLimit', 30) # time limit
model1.model.setParam('PoolSolutions', 1)
model1.model.setParam('Cuts', 0)
model1.model.setParam('CutPasses', 0)
model1.model.setParam('Heuristics', 0)
model1.model.setParam('Symmetry', 0)
model1.model.setParam('Threads', 1)
model1.model.setParam('Presolve', 0)

print('Running simulated annealing...')
S, SAtime = model2.simulated_annealing(
    T=5e5,
    c=0.99,
    L=1,
    min_T=1,
    max_time=SAtimelimit
)
model2.set_solution(S)

model2.set_constraints()
model2.set_objective()

model2.set_valid_inequalities()
model2.set_cutting_planes()

results2 = model2.optimize()


print()
print('Final Summary')
print()

print('-----------------------------------')
print('MODEL 0')
print(f'Status: {model0.get_status()}')
print(f'Runtime: {results0["runtime"]:.2f}')
print(f'Gap: {results0["gap"]*100:.2f}%')
print(f'Objective value: {results0["obj"]:.2e}')

print('-----------------------------------')
print('MODEL 1')
print(f'Status: {model1.get_status()}')
print(f'Simulated Annealing time: {SAtime:.2f}s')
print(f'Runtime: {results1["runtime"]:.2f}s')
print(f'Total time: {SAtime + results1["runtime"]:.2f}s')
print(f'Gap: {results1["gap"]*100:.2f}%')
print(f'Objective value: {results1["obj"]:.2e}')

print('-----------------------------------')
print('MODEL 2')
print(f'Status: {model2.get_status()}')
print(f'Simulated Annealing time: {SAtime:.2f}s')
print(f'Runtime: {results2["runtime"]:.2f}s')
print(f'Total time: {SAtime + results2["runtime"]:.2f}s')
print(f'Gap: {results2["gap"]*100:.2f}%')
print(f'Objective value: {results2["obj"]:.2e}')