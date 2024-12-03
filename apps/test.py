from railway import *

# Constant values
n = 10
periods = 30
jobs = 6
passengers = 100
K = 3

# Create model
model = Railway(n, periods, jobs, passengers, K)

# Generate problem parameters
pi_min_time = 1
pi_max_time = periods // jobs
Aj_min_length = 1
Aj_max_length = n // 3
tau_min_interval = 0
tau_max_interval = 0
phi_min_demand = 0
phi_max_demand = passengers
beta_min_share = 0.5
beta_max_share = 0.7
Lambd_min_capacity = int(0.5 * passengers/n)
Lambd_max_capacity = int(0.7 * passengers/n)
n_max_events = 0
E_min_length = 1
E_max_length = None

model.generate_problem(
    pi_min_time,
    pi_max_time,
    Aj_min_length,
    Aj_max_length,
    tau_min_interval,
    tau_max_interval,
    phi_min_demand,
    phi_max_demand,
    beta_min_share,
    beta_max_share,
    Lambd_min_capacity,
    Lambd_max_capacity,
    n_max_events,
    E_min_length,
    E_max_length,
)

# Set constraints
model.set_constraints()

# Set objective
model.set_objective()

# Solve the problem
model.optimize()