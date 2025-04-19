from railway import *

# Load model

# Define problem parameters
# N = 10 # 10, 20, 40
# J = 80 # 10, 40, 80
# T = 10 # 10, 50, 100
P = 2000
K = 3
timelimit = 60

# N = 10; J = 10; T = 10  #    1
# N = 10; J = 10; T = 50  #    2
# N = 10; J = 10; T = 100 #    3
# N = 10; J = 40; T = 10  #    4
# N = 10; J = 40; T = 50  #    5
# N = 10; J = 40; T = 100 #    6
N = 10; J = 80; T = 10  #    7
# N = 10; J = 80; T = 50  #    8
# N = 10; J = 80; T = 100 #    9
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

# Instantiate the Railway object
model = Railway.load(FILENAME)


# Set up the three models by (deep) copying the blueprint model
model0 = Railway.copy(model)
model1 = Railway.copy(model)
# model2 = Railway.copy(model)



# Model 0: "as-is" Gurobi model with no heuristics or cuts
print("\nModel 0\n")
# model0.model.setParam('OutputFlag', 0) # verbose
model0.model.setParam('TimeLimit', timelimit) # time limit
model0.model.setParam('LPWarmStart', 0)
model0.model.setParam('PoolSolutions', 1)
model0.model.setParam('Cuts', 0)
model0.model.setParam('CutPasses', 0)
model0.model.setParam('Heuristics', 0)
model0.model.setParam('Symmetry', 0)
model0.model.setParam('Threads', 1)
model0.model.setParam('Presolve', 1)



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
# model1.model.setParam('OutputFlag', 0) # verbose
model1.model.setParam('TimeLimit', timelimit) # time limit
model1.model.setParam('LPWarmStart', 0)
model1.model.setParam('PoolSolutions', 1)
model1.model.setParam('Cuts', 0)
model1.model.setParam('CutPasses', 0)
model1.model.setParam('Heuristics', 0)
model1.model.setParam('Symmetry', 0)
model1.model.setParam('Threads', 1)
model1.model.setParam('Presolve', 1)

print('Running simulated annealing...')
S, SAtime = model1.simulated_annealing()
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



# # Model 2: Full model plus valid inequalities and cutting planes
# print("\nModel 2\n")
# # model2.model.setParam('OutputFlag', 0) # verbose
# model2.model.setParam('TimeLimit', 30) # time limit

# S = model2.simulated_annealing()
# model2.set_solution(S)

# model2.set_constraints()
# model2.set_objective()

# model2.set_valid_inequalities()
# model2.set_cutting_planes()

# results2 = model2.optimize()
# # print("Runtime:", results2['runtime'])
# # print("Gap:", results2['mip_gap'])
# # print("Objective value:", results2['obj_val'])


print()
print('Final Summary')
print()

print('-----------------------------------')
print('MODEL 0')
print(f'Status: {model0.get_status()}')
print("Runtime:", results0['runtime'])
print("Gap:", results0['gap'])
print("Objective value:", results0['obj'])

print('-----------------------------------')
print('MODEL 1')
print(f'Status: {model1.get_status()}')
print("Simulated Annealing time:", SAtime)
print("Runtime:", results1['runtime'])
print("Total time:", SAtime + results1['runtime'])
print("Gap:", results1['gap'])
print("Objective value:", results1['obj'])