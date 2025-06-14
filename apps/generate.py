import os
import sys
from railway import *

# Define range of values for sets
N_range = [10, 20, 40]
T_range = [10, 50, 100]
J_range = [10, 40, 80]
# E_range = [] <-- in toy example empty...
# C_range = [] <-- in toy example empty...

# Define constant values for parameters
TIME_LIMIT = 30
passengers = 1000
K = 3

# Check that the script is running from root directory
APPS_DIR = 'apps'
if not os.path.exists(APPS_DIR):
    print('Please run this script from the root directory')
    sys.exit(1)

# Create dataset directory if it does not exist
DATASET_DIR = 'datasets'
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# Create feasible toy instances datasets for the problem
for stations in N_range:
    for periods in T_range:
        for jobs in J_range:

            created = False
            while not created:
            
                # Create the model for the scheduling problem
                rail = Railway(stations, periods, jobs, passengers, K)
                rail.model.setParam('TimeLimit', TIME_LIMIT)

                # Define variable values of the parameters
                job_min_time, job_max_time = 1, periods//2
                job_min_length, job_max_length = 1, 1
                pause_min_time, pause_max_time = 0, 2
                min_demand, max_demand = 0.0, 1.0
                min_share, max_share = 0.1, 0.9
                min_capacity, max_capacity = 0.5, 0.75
                n_max_events = 0
                event_min_length, event_max_length = 0, 0
                
                # Generate the problem
                rail.generate(
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

                # Solve the problem and check that is feasible before saving it
                rail.set_constraints()
                rail.set_objective()
                rail.optimize()
    
                if (rail.get_status() != "INFEASIBLE"
                    and rail.get_status() != "UNBOUNDED"
                    and rail.get_status() != "INF_OR_UNBD"
                    and rail.get_status() != "CUTOFF"
                    and rail.get_status() != "NUMERIC"):
                    created = True
                    FILENAME = os.path.join(DATASET_DIR, f"railway_N{stations}_T{periods}_J{jobs}_P{passengers}_K{K}.json")
                    rail.save(FILENAME)	
                    print(f"✅ Dataset N={stations}, T={periods}, J={jobs}, P={passengers}, K={K} created")
                    del rail