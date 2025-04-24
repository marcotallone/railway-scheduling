from railway import *
import pandas as pd

# Load model

# Define problem parameters
P = 2000
K = 3
timelimit = 300
heuristics_timelimit = 60

# N = 10; J = 10; T = 10 ; ID =  1
# N = 10; J = 10; T = 50 ; ID =  2
# N = 10; J = 10; T = 100; ID =  3
# N = 10; J = 40; T = 10 ; ID =  4
# N = 10; J = 40; T = 50 ; ID =  5
N = 10; J = 40; T = 100; ID =  6
# N = 10; J = 80; T = 10 ; ID =  7
# N = 10; J = 80; T = 50 ; ID =  8
# N = 10; J = 80; T = 100; ID =  9
# N = 20; J = 10; T = 10 ; ID = 10
# N = 20; J = 10; T = 50 ; ID = 11
# N = 20; J = 10; T = 100; ID = 12
# N = 20; J = 40; T = 10 ; ID = 13
# N = 20; J = 40; T = 50 ; ID = 14
# N = 20; J = 40; T = 100; ID = 15
# N = 20; J = 80; T = 10 ; ID = 16
# N = 20; J = 80; T = 50 ; ID = 17
# N = 20; J = 80; T = 100; ID = 18
# N = 40; J = 10; T = 10 ; ID = 19
# N = 40; J = 10; T = 50 ; ID = 20
# N = 40; J = 10; T = 100; ID = 21
# N = 40; J = 40; T = 10 ; ID = 22
# N = 40; J = 40; T = 50 ; ID = 23
# N = 40; J = 40; T = 100; ID = 24
# N = 40; J = 80; T = 10 ; ID = 25
# N = 40; J = 80; T = 50 ; ID = 26
# N = 40; J = 80; T = 100; ID = 27

# Name of the file to load
FILENAME = f"datasets/railway_N{N}_T{T}_J{J}_P{P}_K{K}.json"

# Display problem parameters
print('Time limit:', timelimit)
print('Heuristics time limit:', heuristics_timelimit)
print(f'ID: {ID}, N: {N}, J: {J}, T: {T}, P: {P}, K: {K}')

# Create a csv file for scalability results if it doesn't exist yet
RESULTFILE = f"apps/results.csv"
columns=[
    'ID',
    'model',
    'N',
    'J',
    'T',
    'P',
    'K',
    'status',
    'runtime',
    'heuristics_time',
    'total_time',
    'gap',
    'objective',
    'nodes',
    'iterations',
    'timelimit',
    'heuristics_limit'
]
try:
    df = pd.read_csv(RESULTFILE)
except FileNotFoundError:
    df = pd.DataFrame( columns=columns)
    df.to_csv(RESULTFILE, index=False)
    print(f"Created {RESULTFILE} file for scalability results.")


# Model 0: "as-is" Gurobi model with no heuristics or cuts
# print("\nModel 0\n")
model0 = Railway.load(FILENAME)
model0.model.setParam('OutputFlag', 0) # verbose
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
row0 = {
    'ID': ID,
    'model': 0,
    'N': N,
    'J': J,
    'T': T,
    'P': P,
    'K': K,
    'status': model0.get_status(),
    'runtime': results0['runtime'],
    'heuristics_time': 0,
    'total_time': results0['runtime'],
    'gap': results0['gap'],
    'objective': results0['obj'],
    'nodes': results0['nodes'],
    'iterations': results0['iterations'],
    'timelimit': timelimit,
    'heuristics_limit': heuristics_timelimit
}

# Live results
print()
print(
    f" Model".ljust(6),
    f"| ID".ljust(4),
    f"| status".ljust(13),
    f"| runtime".ljust(10),
    f"| heuristics".ljust(12),
    f"| total".ljust(10),
    f"| gap".ljust(10),
    f"| objective".ljust(12),
    f"| nodes".ljust(10),
)
print("-" * 88)

print(
    f"   {0}".ljust(6),
    f"| {ID}".ljust(4),
    f"| {model0.get_status()}".ljust(13),
    f"| {results0['runtime']:.2f}".ljust(10),
    f"| {0:.2f}".ljust(12),
    f"| {results0['runtime']:.2f}".ljust(10),
    f"| {100*results0['gap']:.2f}%".ljust(10),
    f"| {results0['obj']:.2e}".ljust(12),
    f"| {results0['nodes']}".ljust(10),
)


# Model 1: model with simulated annealing heuristic
# print("\nModel 1\n")
model1 = Railway.load(FILENAME)
model1.model.setParam('OutputFlag', 0) # verbose
model1.model.setParam('TimeLimit', timelimit) # time limit
# model1.model.setParam('LPWarmStart', 1) # Must use warm start from SA
model1.model.setParam('PoolSolutions', 1)
model1.model.setParam('Cuts', 0)
model1.model.setParam('CutPasses', 0)
model1.model.setParam('Heuristics', 0)
model1.model.setParam('Symmetry', 0)
model1.model.setParam('Threads', 1)
model1.model.setParam('Presolve', 0)

# print('Running simulated annealing...')
S, SAtime = model1.simulated_annealing(
    T=5e5,
    c=0.99,
    L=1,
    min_T=1,
    max_time=heuristics_timelimit
)
model1.set_solution(S)

model1.set_constraints()
model1.set_objective()

results1 = model1.optimize()
row1 = {
    'ID': ID,
    'model': 1,
    'N': N,
    'J': J,
    'T': T,
    'P': P,
    'K': K,
    'status': model1.get_status(),
    'runtime': results1['runtime'],
    'heuristics_time': SAtime,
    'total_time': SAtime + results1['runtime'],
    'gap': results1['gap'],
    'objective': results1['obj'],
    'nodes': results1['nodes'],
    'iterations': results1['iterations'],
    'timelimit': timelimit,
    'heuristics_limit': heuristics_timelimit
}

print(
    f"   {1}".ljust(6),
    f"| {ID}".ljust(4),
    f"| {model1.get_status()}".ljust(13),
    f"| {results1['runtime']:.2f}".ljust(10),
    f"| {SAtime:.2f}".ljust(12),
    f"| {SAtime + results1['runtime']:.2f}".ljust(10),
    f"| {100*results1['gap']:.2f}%".ljust(10),
    f"| {results1['obj']:.2e}".ljust(12),
    f"| {results1['nodes']}".ljust(10),
)


# Model 2: Full model plus valid inequalities and cutting planes
# print("\nModel 2\n")
model2 = Railway.load(FILENAME)
model2.model.setParam('OutputFlag', 0) # verbose
model2.model.setParam('TimeLimit', timelimit) # time limit
model1.model.setParam('PoolSolutions', 1)
model1.model.setParam('Cuts', 0)
model1.model.setParam('CutPasses', 0)
model1.model.setParam('Heuristics', 0)
model1.model.setParam('Symmetry', 0)
model1.model.setParam('Threads', 1)
model1.model.setParam('Presolve', 0)

# print('Running simulated annealing...')
S, SAtime = model2.simulated_annealing(
    T=5e5,
    c=0.99,
    L=1,
    min_T=1,
    max_time=heuristics_timelimit
)
model2.set_solution(S)

model2.set_constraints()
model2.set_objective()

model2.set_valid_inequalities()
model2.set_cutting_planes()

results2 = model2.optimize()
row2 = {
    'ID': ID,
    'model': 2,
    'N': N,
    'J': J,
    'T': T,
    'P': P,
    'K': K,
    'status': model2.get_status(),
    'runtime': results2['runtime'],
    'heuristics_time': SAtime,
    'total_time': SAtime + results2['runtime'],
    'gap': results2['gap'],
    'objective': results2['obj'],
    'nodes': results2['nodes'],
    'iterations': results2['iterations'],
    'timelimit': timelimit,
    'heuristics_limit': heuristics_timelimit
}

print(
    f"   {2}".ljust(6),
    f"| {ID}".ljust(4),
    f"| {model2.get_status()}".ljust(13),
    f"| {results2['runtime']:.2f}".ljust(10),
    f"| {SAtime:.2f}".ljust(12),
    f"| {SAtime + results2['runtime']:.2f}".ljust(10),
    f"| {100*results2['gap']:.2f}%".ljust(10),
    f"| {results2['obj']:.2e}".ljust(12),
    f"| {results2['nodes']}".ljust(10),
)

# Append results to the dataframe
df.loc[len(df)] = row0
df.loc[len(df)] = row1
df.loc[len(df)] = row2

# Save the dataframe to the csv file
df.to_csv(RESULTFILE, index=False)