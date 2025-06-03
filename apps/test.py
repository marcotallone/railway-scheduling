# test.py
# Script containing all implemented models and algorithms and testing them
# on small istances of the railway problem.

# Import railway class
from railway import *

# Define problem parameters
N = 10      # Number of stations
J = 10      # Number of jobs
T = 50      # Number of time periods
P = 1000    # Average number of passengers
K = 3       # Number of alternative routes
timelimit = 60
SAtimelimit = 30
verbose = False # set 'True' if you wish to see Gurobi verbose output

# Name of the file to load
FILENAME = f"datasets/railway_N{N}_T{T}_J{J}_P{P}_K{K}.json"

# (Optional: willingly you can delete the next 3 lines...)
# Start a dummy model just to trigger license information notice message
dummy_model = Railway(10, 10, 10, P, K)
del dummy_model

# Display problem informations
print(f'\n\nSelected problem with:')
print(f'  - {N} stations')
print(f'  - {J} jobs')
print(f'  - {T} time periods')
print(f'  - {P} passengers per arc')
print(f'  - {K} alternative routes')

# Model 0: "as-is" Gurobi model with no heuristics or cuts
print("\n\nSolving with Model 0")
model0 = Railway.load(FILENAME)
model0.set_model0(timelimit, verbose)   # set parameters, timelimit and verbose
model0.set_constraints()                # set constraints
model0.set_objective()                  # set objective

results0 = model0.optimize()

print('--------------------')
print('Runnning Gurobi...')
print(f'Status: {model0.get_status()}')
print("Runtime:", results0['runtime'])
print("Gap:", results0['gap'])
print("Objective value:", results0['obj'])



# Model 1: model with simulated annealing heuristic
print("\n\nSolving with Model 1")
model1 = Railway.load(FILENAME)
model1.set_model1(timelimit, verbose)

print('--------------------')
print('Running simulated annealing (SA)...')
S1, SAtime1 = model1.simulated_annealing(
    T=5e3,
    c=0.99,
    L=1,
    min_T=1,
    max_time=SAtimelimit,
    debug=True
)
model1.set_solution(S1)

# Obtain objective value from heuristic solution
_, _, _, _, v1 = model1.get_vars_from_times(S1)
SAobj1 = model1.get_objective_value(v1)

model1.set_constraints()
model1.set_objective()

results1 = model1.optimize()

print('--------------------')
print('Runnning Gurobi...')
print(f'Status: {model1.get_status()}')
print("Simulated Annealing time:", SAtime1)
print("Runtime:", results1['runtime'])
print("Total time:", SAtime1 + results1['runtime'])
print("Gap:", results1['gap'])
print(f'SA Solution: {SAobj1}')
print("Objective value:", results1['obj'])



# Model 2: Full model plus valid inequalities and cutting planes
print("\n\nSolving with Model 2")
model2 = Railway.load(FILENAME)
model2.set_model2(timelimit, verbose)

print('--------------------')
print('Running simulated annealing (SA)...')
S2, SAtime2 = model2.simulated_annealing(
    T=5e3,
    c=0.99,
    L=1,
    min_T=1,
    max_time=SAtimelimit,
    debug=True
)
model2.set_solution(S2)

# Obtain objective value from heuristic solution
_, _, _, _, v2 = model2.get_vars_from_times(S2)
SAobj2 = model2.get_objective_value(v2)

model2.set_constraints()
model2.set_objective()

model2.set_valid_inequalities()
model2.set_cutting_planes()

results2 = model2.optimize()

print('Runnning Gurobi...')
print(f'Status: {model2.get_status()}')
print("Simulated Annealing time:", SAtime2)
print("Runtime:", results2['runtime'])
print("Total time:", SAtime2 + results2['runtime'])
print("Gap:", results2['gap'])
print(f'SA Solution: {SAobj2}')
print("Objective value:", results2['obj'])



print('\n')
print('-----------------------------------')
print('| Final Summary                   |')
print('-----------------------------------')
print('MODEL 0')
print(f'Status: {model0.get_status()}')
print(f'Runtime: {results0["runtime"]:.2f}')
print(f'Gap: {results0["gap"]*100:.2f}%')
print(f'Objective value: {results0["obj"]:.4e}')

print('-----------------------------------')
print('MODEL 1')
print(f'Status: {model1.get_status()}')
print(f'Simulated Annealing time: {SAtime1:.2f}s')
print(f'Runtime: {results1["runtime"]:.2f}s')
print(f'Total time: {SAtime1 + results1["runtime"]:.2f}s')
print(f'Gap: {results1["gap"]*100:.2f}%')
print(f'SA Solution: {SAobj1:.4e}')
print(f'Objective value: {results1["obj"]:.4e}')

print('-----------------------------------')
print('MODEL 2')
print(f'Status: {model2.get_status()}')
print(f'Simulated Annealing time: {SAtime2:.2f}s')
print(f'Runtime: {results2["runtime"]:.2f}s')
print(f'Total time: {SAtime2 + results2["runtime"]:.2f}s')
print(f'Gap: {results2["gap"]*100:.2f}%')
print(f'SA Solution: {SAobj2:.4e}')
print(f'Objective value: {results2["obj"]:.4e}')
