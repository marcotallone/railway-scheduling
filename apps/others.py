# Imports
import gurobipy as gp
from gurobipy import GRB
import ast
import os
import itertools as it
import time 

# Model 1o
def optimization0(
    filename,
    # T, 
    # N,
    # N_coordinates,
    # A,
    # w_exp,
    # w_inc,
    # OD,
    # phi,
    # R,
    # W_exp,
    # J,
    # J_a,
    # p,
    # E,
    # C,
    # t_int,
    runtime,
):

    # Gurobi model
    model = gp.Model()

    # Load and set parameters
    open_parameters = []
    with open(filename, "r") as f:
        lines = f.readlines()
        counter = 0
        for line in lines:
            if counter > 0:
                open_parameters.append(ast.literal_eval(line))
            counter += 1
    T,N, N_coordinates, A, w_exp, w_inc, OD, phi,R, W_exp,J,J_a,p, E, C, t_int  = open_parameters

    lambda_e = 0
    k = 3
    M = 100_000
    
    # Search for arcs with no jobs
    arc_no_job = []
    for a in A:
        if len(J_a[a]) == 0:        
            arc_no_job.append(a)
    
    # Search for never used paths
    h_irrel = []
    for (o,d) in OD:
        max_len = {}
        min_len = {}
        for i in range(1,k+1):
            #compute max and min traveltime
            max_len[i] = sum(w_exp[a] if a in arc_no_job else w_inc[a] for a in R[o,d,i])
            min_len[i] = sum(w_exp[a] for a in R[o,d,i])        
        
        for comb in it.combinations(range(1,k+1),2):
            if max_len[comb[0]] < min_len[comb[1]]:
                h_irrel.append((o,d,comb[1]))
            elif max_len[comb[1]] < min_len[comb[0]]:
                h_irrel.append((o,d,comb[0]))
            
    #threshold checking:
    #simple variant without the conversion from track segments to arcs:
    arc_available = []
    for t in range(1,T+1):
        for a in E[t]: #for each track segment in event request area at time t
            if len(J_a[a]) > 0: #if a job could be scheduled
                min_phi = 0
                #check if there is an od pair that certainly uses that arc
                for (o,d) in OD: 
                    if a in R[o,d,1] and a in R[o,d,2] and a in R[o,d,3]:
                        min_phi += phi[o,d,t] #multiply with factor
                        
                if min_phi >= lambda_e:
                    arc_available.append((a,t))
                        

    # Setup
    model_variables = True
    model_constraints = True
    model_objective = True
    model_run = True
    starttime = time.time()
    

    # Variables
    if model_variables:

        y = model.addVars(J,range(1,T+1),vtype=gp.GRB.BINARY, name='y')
        
        x = model.addVars(A,range(1,T+1),vtype=gp.GRB.BINARY, name='x')
        
        h = model.addVars(OD,range(1,T+1),range(1,k+1),vtype=gp.GRB.BINARY, name='h')

        w = model.addVars(A,range(1,T+1),vtype=gp.GRB.CONTINUOUS, name='w')
        
        v = model.addVars(OD,range(1,T+1),vtype=gp.GRB.CONTINUOUS, name='v')
        
    
    # Constraints
    if model_constraints:

        # (2)
        model.addConstrs(
            sum(y[j,t] for t in range(1, T - p[j] + 1)) 
            == 1
            for j in J
        )

        # (3)
        model.addConstrs(
            x[s_1,t_1,t] 
            + sum(
                y[j,t_2] 
                for t_2 in range(max(1,t-p[j]+1),min(t,T)+1)
            ) 
            <= 1  
            for (s_1,t_1) in A 
            for j in J_a[s_1,t_1]  
            for t in range(1,T+1)
        )
    
        # (5)
        model.addConstrs(
            sum(x[s_1,t_1,t] for t in range(1, T+1)) 
            == T -  sum(p[j] for j in J_a[s_1,t_1])
            for (s_1,t_1) in A
        ) 
        
        # (9)
        model.addConstrs(
            sum(h[o,d,t,i] for i in range(1,k+1)) == 1
            for (o,d) in OD
            for t in range(1,T+1)
        )

        # (15)
        model.addConstrs(h[o,d,t,i] == 0 for (o,d,i) in h_irrel for t in range(1,T+1) )
        
        # (10)
        model.addConstrs(
            v[o,d,t] 
            >= sum(w[s_1,t_1,t] for (s_1,t_1) in R[o,d,i]) 
            - M*(1-h[o,d,t,i]) 
            for i in range(1,k+1) 
            for (o,d) in OD 
            for i in range(1,k+1) 
            for t in range(1,T+1)
        )
        
        # (11)
        model.addConstrs(
            v[o,d,t] 
            <= sum(w[s_1,t_1,t] for (s_1,t_1) in R[o,d,i]) 
            for i in range(1,k+1)
            for (o,d) in OD
            for i in range(1,k+1)
            for t in range(1,T+1)
        )
    
        # (4)
        model.addConstrs(
            w[s_1,t_1,t] 
            == w_exp[s_1,t_1]*x[s_1,t_1,t] 
            + w_inc[s_1,t_1]*(1-x[s_1,t_1,t]) 
            for (s_1,t_1) in A 
            for t in range(1,T+1)
        )
        
        # (13)
        model.addConstrs(
            w[s_1,t_1,t] == w_exp[s_1,t_1] 
            for (s_1,t_1) in arc_no_job 
            for t in range(1,T+1)
        )
    
        # (8)
        model.addConstrs(
            sum(
                sum(
                    h[o,d,t,i]*phi[o,d,t] 
                    for i in range(1,k+1) 
                    if (s_1,t_1) in R[o,d,i]
                ) 
                for (o,d) in OD
            )
            <= lambda_e + x[s_1,t_1,t]*M  
            for t in range(1,T+1) 
            for (s_1,t_1) in E[t]
        ) 

        # (12) + (14) bcause arc_available is a bigger set of arc_no_job
        model.addConstrs(
            x[arc[0],arc[1],t] == 1  
            for (arc,t) in arc_available
        )

        # (6)
        # model.addConstrs(
        #     sum((1 - x[s_1,t_1,t]) for (s_1,t_1) in c) <= 1 
        #     for c in C 
        #     for t in range(1,T+1)
        # )
        
        # (7)
        model.addConstrs(
            sum(
                sum(
                    y[j,t_1] 
                    for t_1 in range(max(1,t - p[j] - t_int +1),t+1)
                )
                for j in J_a[a]
            )
            <= 1
            for t in range(1,T+1) 
            for a in A
        )
        
    
    # Objective function
    if model_objective:
        obj = sum(sum((phi[o,d,t]*(v[o,d,t] - W_exp[o,d])) for t in range(1,T+1)) for (o,d) in OD) 
        model.setObjective(obj,GRB.MINIMIZE)


    # Run & Optimize
    if model_run:
        model.setParam('OutputFlag', 0) # verbose
        model.setParam('TimeLimit',runtime)
        model.setParam('LPWarmStart',0)
        model.setParam('PoolSolutions', 1)
        model.params.presolve = 0
        model.params.cuts = 0
        model.params.cutpasses = 0
        model.params.threads = 1
        model.params.heuristics = 0
        model.params.symmetry = 0
        
        model.optimize()
        endtime = time.time()
        # model.printQuality()

    return (
        model.Status,
        model.Runtime,
        0.0,
        endtime - starttime,
        model.MIPgap,
        model.ObjVal,
        model.NodeCount,
        model.IterCount
    )


# Model 1m
def optimization0(
    T, 
    N,
    N_coordinates,
    A,
    w_exp,
    w_inc,
    OD,
    phi,
    R,
    W_exp,
    J,
    J_a,
    p,
    E,
    C,
    t_int,
    runtime,
):

    # Gurobi model
    model = gp.Model()

    lambda_e = 0
    k = 3
    M = 100_000
    
    # Search for arcs with no jobs
    arc_no_job = []
    for a in A:
        if len(J_a[a]) == 0:        
            arc_no_job.append(a)
    
    # Search for never used paths
    h_irrel = []
    for (o,d) in OD:
        max_len = {}
        min_len = {}
        for i in range(1,k+1):
            #compute max and min traveltime
            max_len[i] = sum(w_exp[a] if a in arc_no_job else w_inc[a] for a in R[o,d,i])
            min_len[i] = sum(w_exp[a] for a in R[o,d,i])        
        
        for comb in it.combinations(range(1,k+1),2):
            if max_len[comb[0]] < min_len[comb[1]]:
                h_irrel.append((o,d,comb[1]))
            elif max_len[comb[1]] < min_len[comb[0]]:
                h_irrel.append((o,d,comb[0]))
            
    #threshold checking:
    #simple variant without the conversion from track segments to arcs:
    arc_available = []
    for t in range(1,T+1):
        for a in E[t]: #for each track segment in event request area at time t
            if len(J_a[a]) > 0: #if a job could be scheduled
                min_phi = 0
                #check if there is an od pair that certainly uses that arc
                for (o,d) in OD: 
                    if a in R[o,d,1] and a in R[o,d,2] and a in R[o,d,3]:
                        min_phi += phi[o,d,t] #multiply with factor
                        
                if min_phi >= lambda_e:
                    arc_available.append((a,t))
                        

    # Setup
    model_variables = True
    model_constraints = True
    model_objective = True
    model_run = True
    starttime = time.time()
    

    # Variables
    if model_variables:

        y = model.addVars(J,range(1,T+1),vtype=gp.GRB.BINARY, name='y')
        
        x = model.addVars(A,range(1,T+1),vtype=gp.GRB.BINARY, name='x')
        
        h = model.addVars(OD,range(1,T+1),range(1,k+1),vtype=gp.GRB.BINARY, name='h')

        w = model.addVars(A,range(1,T+1),vtype=gp.GRB.CONTINUOUS, name='w')
        
        v = model.addVars(OD,range(1,T+1),vtype=gp.GRB.CONTINUOUS, name='v')
        
    
    # Constraints
    if model_constraints:

        # (2)
        model.addConstrs(
            sum(y[j,t] for t in range(1, T - p[j] + 1)) 
            == 1
            for j in J
        )

        # (3)
        model.addConstrs(
            x[s_1,t_1,t] 
            + sum(
                y[j,t_2] 
                for t_2 in range(max(1,t-p[j]+1),min(t,T)+1)
            ) 
            <= 1  
            for (s_1,t_1) in A 
            for j in J_a[s_1,t_1]  
            for t in range(1,T+1)
        )
    
        # (5)
        model.addConstrs(
            sum(x[s_1,t_1,t] for t in range(1, T+1)) 
            == T -  sum(p[j] for j in J_a[s_1,t_1])
            for (s_1,t_1) in A
        ) 
        
        # (9)
        model.addConstrs(
            sum(h[o,d,t,i] for i in range(1,k+1)) == 1
            for (o,d) in OD
            for t in range(1,T+1)
        )

        # (15)
        model.addConstrs(h[o,d,t,i] == 0 for (o,d,i) in h_irrel for t in range(1,T+1) )
        
        # (10)
        model.addConstrs(
            v[o,d,t] 
            >= sum(w[s_1,t_1,t] for (s_1,t_1) in R[o,d,i]) 
            - M*(1-h[o,d,t,i]) 
            for i in range(1,k+1) 
            for (o,d) in OD 
            for i in range(1,k+1) 
            for t in range(1,T+1)
        )
        
        # (11)
        model.addConstrs(
            v[o,d,t] 
            <= sum(w[s_1,t_1,t] for (s_1,t_1) in R[o,d,i]) 
            for i in range(1,k+1)
            for (o,d) in OD
            for i in range(1,k+1)
            for t in range(1,T+1)
        )
    
        # (4)
        model.addConstrs(
            w[s_1,t_1,t] 
            == w_exp[s_1,t_1]*x[s_1,t_1,t] 
            + w_inc[s_1,t_1]*(1-x[s_1,t_1,t]) 
            for (s_1,t_1) in A 
            for t in range(1,T+1)
        )
        
        # (13)
        model.addConstrs(
            w[s_1,t_1,t] == w_exp[s_1,t_1] 
            for (s_1,t_1) in arc_no_job 
            for t in range(1,T+1)
        )
    
        # (8)
        model.addConstrs(
            sum(
                sum(
                    h[o,d,t,i]*phi[o,d,t] 
                    for i in range(1,k+1) 
                    if (s_1,t_1) in R[o,d,i]
                ) 
                for (o,d) in OD
            )
            <= lambda_e + x[s_1,t_1,t]*M  
            for t in range(1,T+1) 
            for (s_1,t_1) in E[t]
        ) 

        # (12) + (14) bcause arc_available is a bigger set of arc_no_job
        model.addConstrs(
            x[arc[0],arc[1],t] == 1  
            for (arc,t) in arc_available
        )

        # (6)
        # model.addConstrs(
        #     sum((1 - x[s_1,t_1,t]) for (s_1,t_1) in c) <= 1 
        #     for c in C 
        #     for t in range(1,T+1)
        # )
        
        # (7)
        model.addConstrs(
            sum(
                sum(
                    y[j,t_1] 
                    for t_1 in range(max(1,t - p[j] - t_int +1),t+1)
                )
                for j in J_a[a]
            )
            <= 1
            for t in range(1,T+1) 
            for a in A
        )
        
    
    # Objective function
    if model_objective:
        obj = sum(sum((phi[o,d,t]*(v[o,d,t] - W_exp[o,d])) for t in range(1,T+1)) for (o,d) in OD) 
        model.setObjective(obj,GRB.MINIMIZE)


    # Run & Optimize
    if model_run:
        model.setParam('OutputFlag', 0) # verbose
        model.setParam('TimeLimit',runtime)
        model.setParam('LPWarmStart',0)
        model.setParam('PoolSolutions', 1)
        model.params.presolve = 0
        model.params.cuts = 0
        model.params.cutpasses = 0
        model.params.threads = 1
        model.params.heuristics = 0
        model.params.symmetry = 0
        
        model.optimize()
        endtime = time.time()
        # model.printQuality()

    return (
        model.Status,
        model.Runtime,
        0.0,
        endtime - starttime,
        model.MIPgap,
        model.ObjVal,
        model.NodeCount,
        model.IterCount
    )