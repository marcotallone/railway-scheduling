from railway import *
import pandas as pd

# Load model

# Define problem parameters
P = 2000
K = 3
timelimit = 120
heuristics_timelimit = 60
problems = {
    # ID: (N, J, T),
    1: (10, 10, 10),
    2: (10, 10, 50),
    3: (10, 10, 100),
    4: (10, 40, 10),
    5: (10, 40, 50),
    6: (10, 40, 100),
    7: (10, 80, 10),
    8: (10, 80, 50),
    9: (10, 80, 100),
    10: (20, 10, 10),
    11: (20, 10, 50),
    12: (20, 10, 100),
    13: (20, 40, 10),
    14: (20, 40, 50),
    15: (20, 40, 100),
    16: (20, 80, 10),
    17: (20, 80, 50),
    18: (20, 80, 100),
    19: (40, 10, 10),
    20: (40, 10, 50),
    21: (40, 10, 100),
    22: (40, 40, 10),
    23: (40, 40, 50),
    24: (40, 40, 100),
    25: (40, 80, 10),
    26: (40, 80, 50),
    27: (40, 80, 100),
}

# Starat a dummy model just to trigger license information notice message
dummy_model = Railway(10, 10, 10, P, K)
del dummy_model
print()

# Create a csv file for scalability results if it doesn't exist yet
RESULTFILE = f"apps/results3.csv"
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
print("-" * 95)

# Solve problems one by one
for ID, (N, J, T) in problems.items():


    # Name of the file to load
    FILENAME = f"datasets3/railway_N{N}_T{T}_J{J}_P{P}_K{K}.json"


    # Model 0: "as-is" Gurobi model with no heuristics or cuts
    model0 = Railway.load(FILENAME)
    model0.set_model0(timelimit, False)
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

    print(
        f"   {0}".ljust(6),
        f"| {ID}".ljust(4),
        f"| {model0.get_status()}".ljust(13),
        f"| {results0['runtime']:.2f}".ljust(10),
        f"| {0:.2f}".ljust(12),
        f"| {results0['runtime']:.2f}".ljust(10),
        f"| {100*results0['gap']:.2f}%".ljust(10),
        f"| {results0['obj']:.2e}".ljust(12),
        f"| {int(results0['nodes']):d}".ljust(10),
    )


    # Model 1: model with simulated annealing heuristic
    model1 = Railway.load(FILENAME)
    model1.set_model1(timelimit, False)
    S, SAtime = model1.simulated_annealing(
        T=5e3,
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
        f"| {int(results1['nodes']):d}".ljust(10),
    )


    # Model 2: Full model plus valid inequalities and cutting planes
    model2 = Railway.load(FILENAME)
    model2.set_model2(timelimit, False)
    S, SAtime = model2.simulated_annealing(
        T=5e3,
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
        f"| {int(results2['nodes']):d}".ljust(10),
    )


    # Append results to the dataframe
    df.loc[len(df)] = row0
    df.loc[len(df)] = row1
    df.loc[len(df)] = row2

    # Save the dataframe to the csv file
    df.to_csv(RESULTFILE, index=False)

    # Delete the models for next iteration
    del model0
    del model1
    del model2