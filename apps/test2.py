from railway import *

# Constant values
n = 30
periods = 20
jobs = 40
passengers = 2000
K = 3

# Create blueprint model
model = Railway(n, periods, jobs, passengers, K)

# Generate problem parameters
job_min_time, job_max_time = 1, 3
job_min_length, job_max_length = 1, 3
pause_min_time, pause_max_time = 0, 1
min_demand, max_demand = 0.5, 0.9
min_share, max_share = 0.5, 0.9
min_capacity, max_capacity = 0.5, 0.7
n_max_events = 0
event_min_length, event_max_length = 0, 0

model.generate(
    job_min_time,
    job_max_time,
    job_min_length,
    job_max_length,
    pause_min_time,
    pause_max_time,
    min_demand,
    max_demand,
    min_share,
    max_share,
    min_capacity,
    max_capacity,
    n_max_events,
    event_min_length,
    event_max_length,
)

# Set up the three models by (deep) copying the blueprint model
model0 = Railway.copy(model)
model1 = Railway.copy(model)
model2 = Railway.copy(model)

# Model 0: "as-is" Gurobi model with no heuristics or cuts
print("\nModel 0\n")
model0.model.setParam('OutputFlag', 0) # verbose
model0.model.setParam('TimeLimit', 30) # time limit
model0.model.setParam('LPWarmStart', 0)
model0.model.setParam('PoolSolutions', 1)
model0.model.setParam('Cuts', 0)
model0.model.setParam('Heuristics', 0)
model0.model.setParam('Symmetry', 0)
model0.model.setParam('Threads', 1)

model0.set_constraints()
model0.set_objective()

results0 = model0.optimize()
print("Runtime:", results0['runtime'])
print("Gap:", results0['mip_gap'])
print("Objective value:", results0['obj_val'])


# Model 1: model with simulated annealing heuristic
print("\nModel 1\n")
model1.model.setParam('OutputFlag', 0) # verbos
model1.model.setParam('TimeLimit', 30) # time limit
model1.model.setParam('LPWarmStart', 0)
model1.model.setParam('PoolSolutions', 1)
model1.model.setParam('Cuts', 0)
model1.model.setParam('Heuristics', 0)
model1.model.setParam('Symmetry', 0)
model1.model.setParam('Threads', 1)

S = model1.simulated_annealing()
model1.set_solution(S)

model1.set_constraints()
model1.set_objective()

results1 = model1.optimize()
print("Runtime:", results1['runtime'])
print("Gap:", results1['mip_gap'])
print("Objective value:", results1['obj_val'])


# Model 2: Full model plus valid inequalities and cutting planes
print("\nModel 2\n")
model2.model.setParam('OutputFlag', 0) # verbose
model2.model.setParam('TimeLimit', 30) # time limit

S = model2.simulated_annealing()
model2.set_solution(S)

model2.set_constraints()
model2.set_objective()

model2.set_valid_inequalities()
model2.set_cutting_planes()

results2 = model2.optimize()
print("Runtime:", results2['runtime'])
print("Gap:", results2['mip_gap'])
print("Objective value:", results2['obj_val'])

