# Imports
import json
import random
from collections import deque

import gurobipy as gb
import numpy as np
from gurobipy import GRB, Model, quicksum


# Railway Class ----------------------------------------------------------------
class Railway:
    """Class modelling the railway scheduling problem.

    Attributes
    ----------
    stations: int (also `n`)
            number of stations
    periods: int (also `Tend`)
            number of discrete time periods
    jobs: int
            number of maintenance jobs to be scheduled
    passengers: int
            total number of passengers
    routes: int (also `K`)
            number of alternative routes for each event track
    **kwargs: dict
            optional arguments including:
                    - sets Aj, C, E, R
                    - parameters pi, tau, phi, beta, Lambd

    Sets
    ----
    N: list
            set of nodes (stations)
    A: list
            set of arcs (direct train connections)
    OD: list
            set of all possible origin o - destination d pairs (where o != d)
    T: list
            set of discrete time periods
    J: list
            set of maintenance jobs
    Aj: dict
            set of arcs for job j in J
    Ja: dict
            set of maintenance jobs on arc a in A
    C: list
            set of arcs that cannot be unavailable simultaneously
    E: dict
            set of segments (i.e. list of arcs) in an event request at time t
    R: dict
            set of k routes from origin o to destination d

    Parameters
    ----------
    pi: dict
            processing times for each job in J
    tau: dict
            minimum time interval between maintainance jobs on arc a in A
    phi: dict
            passenger demand from origin o to destination d at time t
    beta: dict
            peak share of passenger demand from origin o to destination d at time t
    Lambd: dict
            capacity of alternative services for each segment s (i.e. list of arcs) in E

    Other Attributes
    ----------------
    Omega: dict
            average travel time from origin o to destination d
    omega_e: dict
            average travel time by train on arc a in A (euclidean distance)
    omega_j: dict
            increased travel time by train on arc a in A (delay factor * omega_e)
    delay_factor: float (default 1.35)
            increased travel time factor for service replacement
    coords: list
            list of euclidean coordinates for each station (randomly generated in unit circle)
    graph: dict
            graph representation of the railway network with N nodes and A arcs
    M: int
            upper bound representing unlimited capacity for train service
    model: gurobipy.Model
            Gurobi model for the railway scheduling problem
    S: dict
        Dictionary of starting times for jobs in J

    Variables
    ---------
    y: dict, binary variable
            1 if job j in J is started at time t in T, 0 otherwise
    x: dict, binary variable
            1 if arc a in A is available at time t in T, 0 otherwise
    h: dict, binary variable
            1 if route i in R from origin o to destination d is selected at time t in T, 0 otherwise
    w: dict, continuous variable
            travel time on arc a in A at time t in T
    v: dict, continuous variable
            travel time from origin o to destination d at time t in T

    Methods
    -------

    Constructors
    ------------
    __init__(stations, periods, jobs, passengers, routes, **kwargs)
            main constructor for the railway scheduling problem.
    copy(other)
            create a copy of a Railway object.
    load(filename)
            load the problem parameters from a json file with the necessary data

    Setters
    -------
    set_constraints()
            set the constraints for the railway scheduling problem
    set_objective()
            set the objective function for the railway scheduling problem
    set_start(y_start, x_start, h_start, w_start, v_start)
            set the starting values for the decision variables
    set_solution(S=None)
            set the initial solution for the railway scheduling problem
    set_valid_inequalities()
            set valid inequalities for the railway scheduling problem
    set_cutting_planes()
            set cutting planes for the optimization model
            
    Getters
    -------
    get_status()
            return the status of the optimization model
    get_y()
            get the decision variables values for job j in J started at time t in T
    get_x()
            get the decision variables values for arc a in A available at time t in T
    get_h()
            get the decision variables values for route selection i in R from origin o to destination d selected at time t in T
    get_w()
            get the decision variables values for travel time on arc a in A at time t in T
    get_v()
            get the decision variables values for travel time from origin o to destination d at time t in T
    get_vars_from_times(S)
            get the decision variables values from a set S of starting times for each job in J
    get_times_from_vars(y)
            get the starting times S for each job in J from the decision variables values
    get_objective_value(v)
            evaluate the objective function for the railway scheduling problem

    Optimize
    --------
    optimize()
            optimize and solve the railway scheduling problem
    status()
            return the status of the optimization model
    check_feasibility(y, x, h, w, v)
            check the feasibility of a given solution for the railway scheduling
            problem by checking the constraints one by one

    Static Methods
    --------------
    dijkstra(graph, source)
            Dijkstra's algorithm to find the shortest path in a graph
    shortest_arcs(predecessors, start, end)
            return set of arcs of shortest path from start to end
    shortest_nodes(predecessors, start, end)
            return list of nodes of shortest path from start to end
    nodes_to_arcs(nodes)
            convert a list of nodes to a suitable list of arcs in A
    arcs_to_nodes(arcs)
            convert a list of arcs to a suitable list of nodes in N
    YenKSP(graph, source, sink, K)
            Yen's K-shortest paths algorithm to find the K-shortest paths in a graph

    Generators
    ----------
    generate()
            generate problem parameters for the railway scheduling problem
    generate_initial_solution()
            generate an initial solution S for the railway scheduling problem
    generate_neighbor_solution(S)
            generate a neighbor solution N(S) to a given one S for the railway scheduling problem

    Other Methods
    -------------
    save(filename)
            save the problem parameters to a json file

    """

    # Constructors -------------------------------------------------------------

    # Main constructor
    def __init__(self, stations, periods, jobs, passengers, routes, **kwargs):

        # Attributes
        self.n = stations
        self.stations = stations
        self.Tend = periods
        self.periods = periods
        self.jobs = jobs
        self.passengers = passengers
        self.K = routes
        self.routes = routes

        # Sets
        self.N = range(1, self.n + 1)
        self.A = [(i, j) for i in self.N for j in self.N if i < j]
        self.OD = [(o, d) for o in self.N for d in self.N if o != d]
        self.T = range(1, self.periods + 1)
        self.J = range(1, self.jobs + 1)
        self.Aj = kwargs.get("Aj", {})
        self.__set_Ja()  # set the Ja set
        self.C = kwargs.get("C", [])
        self.E = kwargs.get("E", {})
        self.R = kwargs.get("R", {})

        # Parameters
        self.pi = kwargs.get("pi", {})
        self.tau = kwargs.get("tau", {})
        self.phi = kwargs.get("phi", {})
        self.beta = kwargs.get("beta", {})
        self.Lambd = kwargs.get("Lambd", {})

        # Other attributes
        self.coords = kwargs.get("coords", [])
        if not self.coords:
            self.__generate_coords()
        self.delay_factor = 1.35
        self.omega_e = {
            (i, j): np.sqrt(
                (self.coords[i - 1][0] - self.coords[j - 1][0]) ** 2
                + (self.coords[i - 1][1] - self.coords[j - 1][1]) ** 2
            )
            for i, j in self.A
        }
        self.omega_j = {a: self.delay_factor * self.omega_e[a] for a in self.A}
        self.graph = {node: {} for node in self.N}
        for node in self.N:
            for i, j in self.A:
                if i == node:
                    self.graph[node][j] = self.omega_e[(i, j)]
                elif j == node:
                    self.graph[node][i] = self.omega_e[(i, j)]
        self.Omega = {
            (o, d): self.dijkstra(self.graph, o)[0][
                d
            ]  # 0 because we want the distances (1st element)
            for o, d in self.OD
        }
        self.__set_M()  # set the M upper bound
        self.model = Model()
        self.model.ModelSense = gb.GRB.MINIMIZE
        self.S = {}

        # Decision Variables
        self.y = self.model.addVars(self.J, self.T, vtype=gb.GRB.BINARY, name="y")
        self.x = self.model.addVars(self.A, self.T, vtype=gb.GRB.BINARY, name="x")
        self.h = self.model.addVars(
            self.N, self.N, self.T, range(1, self.K + 1), vtype=gb.GRB.BINARY, name="h"
        )
        self.w = self.model.addVars(
            self.A, self.T, lb=0, vtype=gb.GRB.CONTINUOUS, name="w"
        )
        self.v = self.model.addVars(
            self.N, self.N, self.T, lb=0, vtype=gb.GRB.CONTINUOUS, name="v"
        )

    # Copy constructor
    @classmethod
    def copy(cls, other):
        """Create a copy of a Railway object.

        Parameters
        ----------
        other : Railway
                The Railway object to copy.

        Returns
        -------
        model : Railway
                A separate copy of the Railway object.
        """

        return cls(
            other.stations,
            other.periods,
            other.jobs,
            other.passengers,
            other.routes,
            coords=other.coords,
            Aj=other.Aj,
            C=other.C,
            E=other.E,
            R=other.R,
            pi=other.pi,
            tau=other.tau,
            phi=other.phi,
            beta=other.beta,
            Lambd=other.Lambd,
        )

    # Alternative constructor
    @classmethod
    def load(cls, filename):
        """Load the problem parameters from a json file.

        Parameters
        ----------
        filename : str
                The name of the json file containing the problem parameters.

        Returns
        -------
        model : Railway
                The Railway object created from the data in the json file.

        Raises
        ------
        ValueError
                If the file is missing some of the necessary data to create a Railway object.
        """

        # Open the json file and load the data
        with open(filename, "r") as f:
            data = json.load(f)

        # Check that the file contains the necessary data otherwise raise an error
        if not all(
            key in data
            for key in [
                "stations",
                "periods",
                "jobs",
                "passengers",
                "routes",
                "coords",
                "pi",
                "Aj",
                "C",
                "tau",
                "phi",
                "beta",
                "Lambd",
                "E",
                "R",
            ]
        ):
            raise ValueError(
                "The file is missing some of the necessary data to create a Railway object."
            )

        # Convert string keys and values back to their original values
        # since JSON format does not store tuples
        data["coords"] = [tuple(coord) for coord in data["coords"]]
        data["pi"] = {eval(k): v for k, v in data["pi"].items()}
        data["Aj"] = {eval(k): [tuple(a) for a in v] for k, v in data["Aj"].items()}
        # TODO: data["C"] = ... todo the same for C when will be implemented
        data["tau"] = {eval(k): v for k, v in data["tau"].items()}
        data["phi"] = {eval(k): v for k, v in data["phi"].items()}
        data["beta"] = {eval(k): v for k, v in data["beta"].items()}
        data["Lambd"] = {eval(k): v for k, v in data["Lambd"].items()}
        data["E"] = {
            eval(k1): {eval(k2): [tuple(a) for a in v2] for k2, v2 in v1.items()}
            for k1, v1 in data["E"].items()
        }
        data["R"] = {
            eval(k): [[tuple(a) for a in l] for l in v] for k, v in data["R"].items()
        }

        # Create the model object with main constructor
        model = cls(
            data["stations"],
            data["periods"],
            data["jobs"],
            data["passengers"],
            data["routes"],
            coords=data["coords"],
            Aj=data["Aj"],
            C=data["C"],
            E=data["E"],
            R=data["R"],
            pi=data["pi"],
            tau=data["tau"],
            phi=data["phi"],
            beta=data["beta"],
            Lambd=data["Lambd"],
        )
        return model

    # Operators ----------------------------------------------------------------

    # String representation
    def __str__(self):
        """Return a string representation of the railway scheduling problem."""
        return (
            f"Railway scheduling problem\n"
            f"\n"
            f"Parameters:\n"
            f"N:  {self.n} stations\n"
            f"T:  {self.periods} periods\n"
            f"J:  {self.jobs} jobs\n"
            f"P:  {self.passengers} passengers\n"
            f"K:  {self.routes} alternative routes\n"
            f"Aj: {len(self.Aj)} jobs with arcs\n"
            f"Ja: {len([aj for aj, ja in self.Ja.items() if ja])} arcs with jobs\n"
            f"C:  {len(self.C)} arcs unavailable simultaneously\n"
            f"\n"
            f"Optimization model:\n"
            f"Variables:   {self.model.NumVars}\n"
            f"Constraints: {self.model.NumConstrs}\n"
            f"Objective:   {self.model.getObjective()}\n"
            f"Status:      {self.get_status()}\n"
        )

    # Equality operator
    def __eq__(self, other):
        """Check if two railway scheduling problems are equal."""
        if not isinstance(other, Railway):
            return False
        return (
            self.n == other.n
            and self.periods == other.periods
            and self.jobs == other.jobs
            and self.passengers == other.passengers
            and self.routes == other.routes
            and self.coords == other.coords
            and self.pi == other.pi
            and self.Aj == other.Aj
            and self.Ja == other.Ja
            and self.C == other.C
            and self.tau == other.tau
            and self.phi == other.phi
            and self.beta == other.beta
            and self.Lambd == other.Lambd
            and self.E == other.E
            and self.R == other.R
        )

    # Setters ------------------------------------------------------------------

    # Set Ja set
    def __set_Ja(self):
        if self.Aj == {}:
            self.Ja = {}
        else:
            self.Ja = {a: [j for j in self.J if a in self.Aj[j]] for a in self.A}

    # Set M upper bound
    def __set_M(self):
        if self.phi == {}:
            self.M = 100 * self.passengers
        else:
            self.M = sum(self.phi[(o, d, t)] for o, d in self.OD for t in self.T) + (
                100 * self.passengers
            )

    # Set constraints
    def set_constraints(self):
        """Set the constraints for the railway scheduling problem."""

        # Remove any existing constraints
        self.model.remove(self.model.getConstrs())

        # Job started once and completed within time horizon                    (2)
        self.model.addConstrs(
            (
                quicksum(self.y[j, t] for t in range(1, len(self.T) - self.pi[j] + 1))
                == 1
                for j in self.J
            ),
            name="2",
        )

        # Availability of arc a at time t                                       (3)
        self.model.addConstrs(
            (
                self.x[*a, t]
                + quicksum(
                    self.y[j, tp]
                    for tp in range(max(1, t - self.pi[j] + 1), min(t, self.Tend) + 1)
                )
                <= 1
                for a in self.A
                for t in self.T
                for j in self.Ja[a]
            ),
            name="3",
        )

        # Increased travel time for service replacement                         (4)
        self.model.addConstrs(
            (
                self.w[*a, t]
                == self.x[*a, t] * self.omega_e[a]
                + (1 - self.x[*a, t]) * self.omega_j[a]
                for a in self.A
                for t in self.T
            ),
            name="4",
        )

        # Ensure correct arc travel times for free variables                    (5)
        self.model.addConstrs(
            (
                quicksum(self.x[*a, t] for t in self.T)
                == self.Tend - quicksum(self.pi[j] for j in self.Ja[a])
                for a in self.A
            ),
            name="5",
        )

        # Arcs that cannt be unavailable simultaneously                         (6)
        self.model.addConstrs(
            (
                quicksum(1 - self.x[*a, t] for a in c) <= 1
                for t in self.T
                for c in self.C
                # for j in self.J # TODO: add this only if C depends on job j
            ),
            name="6",
        )

        # Non overlapping jobs on same arc                                      (7)
        self.model.addConstrs(
            (
                quicksum(
                    quicksum(
                        self.y[j, tp]
                        for tp in range(max(1, t - self.pi[j] - self.tau[a] + 1), t + 1)
                    )
                    for j in self.Ja[a]
                )
                <= 1
                for a in self.A
                for t in self.T
            ),
            name="7",
        )

        # Each track segment in an event request has limited capacity           (8)
        self.model.addConstrs(
            (
                quicksum(
                    quicksum(
                        quicksum(
                            self.h[o, d, t, i] * self.beta[o, d, t] * self.phi[o, d, t]
                            for i in range(1, self.K + 1)
                            if a in self.R[(o, d)][i - 1]
                        )
                        for o, d in self.OD
                    )
                    for a in self.E[t][s]
                )
                <= quicksum(self.Lambd[a, t] for a in self.E[t][s])
                + (self.M * quicksum(self.x[*a, t] for a in self.E[t][s]))
                for t in self.T
                for s in self.E[t]
            ),
            name="8",
        )

        # Passeger flow from o to d is served by one of predefined routes       (9)
        self.model.addConstrs(
            (
                quicksum(self.h[o, d, t, i] for i in range(1, self.K + 1)) == 1
                for t in self.T
                for o, d in self.OD
            ),
            name="9",
        )

        # Lower bound for travel time from o to d                               (10)
        self.model.addConstrs(
            (
                self.v[o, d, t]
                >= quicksum(self.w[*a, t] for a in self.R[(o, d)][i - 1])
                - self.M * (1 - self.h[o, d, t, i])
                for t in self.T
                for o, d in self.OD
                for i in range(1, self.K + 1)
            ),
            name="10",
        )

        # Upper bound for travel time from o to d                               (11)
        self.model.addConstrs(
            (
                self.v[o, d, t]
                <= quicksum(self.w[*a, t] for a in self.R[(o, d)][i - 1])
                for t in self.T
                for o, d in self.OD
                for i in range(1, self.K + 1)
            ),
            name="11",
        )

        # Arcs never included in any job are always available                   (12)
        self.model.addConstrs(
            (
                self.x[*a, t] == 1
                for a in self.A
                for t in self.T
                if not any(a in self.Aj[j] for j in self.J)
            ),
            name="12",
        )

        # Travel times for arcs never included in any job are equal to omega_e  (13)
        self.model.addConstrs(
            (
                self.w[*a, t] == self.omega_e[a]
                for a in self.A
                for t in self.T
                if not any(a in self.Aj[j] for j in self.J)
            ),
            name="13",
        )

        # Availability of arcs included in all routes during event requests     (14)
        self.model.addConstrs(
            (
                self.x[*a, t] == 1
                for a in self.A
                for t in self.T
                for s in self.E[t]
                if a in self.E[t][s]
                for o, d in self.OD
                if all(a in self.R[(o, d)][i - 1] for i in range(1, self.K + 1))
                if self.beta[o, d, t]*self.phi[o, d, t] >
                quicksum(self.Lambd[a, t] for a in self.E[t][s])
            ),
            name="14",
        ) 

    # Set objective function
    def set_objective(self):
        """Set the objective function for the railway scheduling problem."""

        # Objective function                                                    (1)
        self.model.setObjective(
            quicksum(
                quicksum(
                    self.phi[o, d, t] * (self.v[o, d, t] - self.Omega[o, d])
                    for t in self.T
                )
                for o, d in self.OD
            )
        )

    # Set variables starting values
    def set_start(self, y_start, x_start, h_start, w_start, v_start):
        """
        Set the starting values for the decision variables.

        Parameters
        ----------
        y_start: dict
            Starting values for y variables.
        x_start: dict
            Starting values for x variables.
        h_start: dict
            Starting values for h variables.
        w_start: dict
            Starting values for w variables.
        """

        # Set the starting values for the decision variables

        for j in self.J:
            for t in self.T:
                if (j, t) in y_start:
                    self.y[j, t].start = y_start[(j, t)]

        for a in self.A:
            for t in self.T:
                if (a, t) in x_start:
                    self.x[*a, t].start = x_start[(a, t)]

        for o, d in self.OD:
                for t in self.T:
                    for k in range(1, self.K + 1):
                        if (o, d, t, k) in h_start:
                            self.h[o, d, t, k].start = h_start[(o, d, t, k)]

        for a in self.A:
            for t in self.T:
                if (a, t) in w_start:
                    self.w[*a, t].start = w_start[(a, t)]

        for o, d in self.OD:
            for t in self.T:
                if (o, d, t) in v_start:
                    self.v[o, d, t].start = v_start[(o, d, t)]

        # Update the model for changes to take effect
        self.model.update()

    # Set solution
    def set_solution(self, S=None):
        """Set the initial solution for the railway scheduling problem
        that will be optimized by the Gurobi solver.

        Parameters
        ----------
        S : dict, optional
                Dictionary of starting times for jobs in J, by default None
                If no input solution is provided the model will check if one
                is stored in the S attribute.

        Raises
        ------
        ValueError
                If no solution S is provided for the railway scheduling problem.
        """

        # Set the solution
        if S is not None:
            y, x, h, w, v = self.get_vars_from_times(S)
            self.set_start(y, x, h, w, v)

        # Check if a solution is available
        elif self.S:
            y, x, h, w, v = self.get_vars_from_times(self.S)
            self.set_start(y, x, h, w, v)

        else:
            raise ValueError(
                """No solution provided for the railway scheduling problem. 
                Please provide a solution or generate one"""
            )             

    # Set valid inequalities
    def set_valid_inequalities(self):
        """Set valid inequalities for the railway scheduling problem.
        The valid inequalities are the Sousa and Wolsey single machine
        scheduling inequalities (B1) and the non overlapping jobs on
        the same arc inequalities (B2).
        """

        # Define the Delta set for (B1)
        def Delta(self, j, a):
            p_times = [ self.pi[jp] + self.tau[a] for jp in self.Ja[a] if jp != j ]
            delta_max = max(2, *p_times) if p_times else 2
            delta_set = range(2, delta_max + 1)
            return delta_set
        
        # Sousa and Wolsey single machine scheduling valid inequalities     (B1)
        self.model.addConstrs(
            (
                quicksum(
                    self.y[j,tp] 
                    for tp in range(t - self.pi[j] + 1, t + delta - 1)  # Qj
                )
                + quicksum(
                    quicksum(
                        self.y[jp,tp]
                        for tp in range(t - self.pi[jp] + delta, t + 1)  # Q'l
                    ) for jp in self.Ja[a] if jp != j
                )
                <= 1
                for a in self.A
                for j in self.Ja[a]
                for t in self.T
                for delta in Delta(self, j, a)
            ),
            name="B1",
        )

        # Non overlapping jobs on same arc                                  (B2)
        self.model.addConstrs(
            (
                self.y[j, t] + self.y[jp, tp] <= 1
                for a in self.A
                for j in self.Ja[a]
                for jp in self.Ja[a] if jp != j
                for t in self.T
                for tp in range(
                        max(1, t - self.pi[jp] - self.tau[a] + 1), 
                        min(t + self.pi[j] + self.tau[a], self.Tend + 1)
                )
            ),
            name="B2",
        )

    # Set cutting planes
    def set_cutting_planes(self):
        """Set cutting planes for the optimization model. The following cutting
        planes are set for the railway scheduling problem:
        
            - Boolean quadratic polytope cuts
            - Clique cuts
            - Cover cuts
            - Flow cover cuts
            - Flow path cuts
            - Gomori cuts
            - GUB cover cuts
            - Implied bound cuts
            - Lift-and-project cuts
            - MIR cuts
            - Mod-k cuts
            - Network cuts
            - Relax-and-lift cuts
            - Strong-CG cuts
            - {0, 1/2} cuts
        """
        
        # Set cutting planes for optimization 
        self.model.params.presolve = 1
        self.model.params.cuts = 0
        self.model.params.BQPCuts = -1
        self.model.params.CliqueCuts = -1
        self.model.params.CoverCuts = -1
        self.model.params.FlowCoverCuts = -1
        self.model.params.FlowPathCuts = -1
        self.model.params.GomoryPasses = -1
        self.model.params.GUBCoverCuts = -1
        self.model.params.ImpliedCuts = -1
        self.model.params.LiftProjectCuts = -1
        self.model.params.MIRCuts = -1
        self.model.params.ModKCuts = -1
        self.model.params.NetworkCuts = -1
        self.model.params.RelaxLiftCuts = -1
        self.model.params.StrongCGCuts = -1
        self.model.params.ZeroHalfCuts = -1
        self.model.params.cutpasses = 0
        # self.model.params.threads = 1
        self.model.params.heuristics = 0
        self.model.params.symmetry = 0

    # Getters ------------------------------------------------------------------

    # Get the status of the optimization model
    def get_status(self):
        """Return the status of the optimization model.

        Returns
        -------
        status : str
                The status of the optimization model, which can be one of the
                following according to Gurobi's documentation:
                - LOADED (1): Model loaded; no solution available.
                - OPTIMAL (2): Solved to optimality; solution available.
                - INFEASIBLE (3): Model proven infeasible.
                - INF_OR_UNBD (4): Model infeasible or unbounded (set DualReductions=0 for clarity).
                - UNBOUNDED (5): Model proven unbounded; feasibility not guaranteed.
                - CUTOFF (6): Objective worse than specified Cutoff; no solution.
                - ITERATION_LIMIT (7): Exceeded iteration limits (simplex/barrier).
                - NODE_LIMIT (8): Exceeded branch-and-cut node limit.
                - TIME_LIMIT (9): Exceeded time limit.
                - SOLUTION_LIMIT (10): Reached solution limit.
                - INTERRUPTED (11): Optimization terminated by user.
                - NUMERIC (12): Terminated due to numerical issues.
                - SUBOPTIMAL (13): Sub-optimal solution available (optimality tolerances not met).
                - INPROGRESS (14): Optimization run not yet complete (asynchronous).
                - USER_OBJ_LIMIT (15): User-defined objective limit reached.
                - WORK_LIMIT (16): Exceeded work limit.
                - MEM_LIMIT (17): Exceeded memory allocation limit.
        """

        # Check the optimization status
        if self.model.status == GRB.Status.LOADED:
            return "LOADED"
        elif self.model.status == GRB.Status.OPTIMAL:
            return "OPTIMAL"
        elif self.model.status == GRB.Status.INFEASIBLE:
            return "INFEASIBLE"
        elif self.model.status == GRB.Status.INF_OR_UNBD:
            return "INF_OR_UNBD"
        elif self.model.status == GRB.Status.UNBOUNDED:
            return "UNBOUNDED"
        elif self.model.status == GRB.Status.CUTOFF:
            return "CUTOFF"
        elif self.model.status == GRB.Status.ITERATION_LIMIT:
            return "ITERATION_LIMIT"
        elif self.model.status == GRB.Status.NODE_LIMIT:
            return "NODE_LIMIT"
        elif self.model.status == GRB.Status.TIME_LIMIT:
            return "TIME_LIMIT"
        elif self.model.status == GRB.Status.SOLUTION_LIMIT:
            return "SOLUTION_LIMIT"
        elif self.model.status == GRB.Status.INTERRUPTED:
            return "INTERRUPTED"
        elif self.model.status == GRB.Status.NUMERIC:
            return "NUMERIC"
        elif self.model.status == GRB.Status.SUBOPTIMAL:
            return "SUBOPTIMAL"
        elif self.model.status == GRB.Status.INPROGRESS:
            return "INPROGRESS"
        elif self.model.status == GRB.Status.USER_OBJ_LIMIT:
            return "USER_OBJ_LIMIT"
        elif self.model.status == GRB.Status.WORK_LIMIT:
            return "WORK_LIMIT"
        elif self.model.status == GRB.Status.MEM_LIMIT:
            return "MEM_LIMIT"
        else:
            return "UNKNOWN"
        
    # Get y decision variables
    def get_y(self):
        """Get the decision variables values for job j in J started at time t in T.

        Returns
        -------
        y : dict
                Dictionary of binary variables for job j in J started at time t in T
        """

        return {(j, t): int(self.y[j, t].x) for j in self.J for t in self.T}
    
    # Get x decision variables
    def get_x(self):
        """Get the decision variables values for arc a in A available at time t in T.

        Returns
        -------
        x : dict
                Dictionary of binary variables for arc a in A available at time t in T
        """

        return {(a, t): int(self.x[*a, t].x) for a in self.A for t in self.T}
    
    # Get h decision variables
    def get_h(self):
        """Get the decision variables values for route selection i in R 
        from origin o to destination d selected at time t in T.

        Returns
        -------
        h : dict
                Dictionary of binary variables for route i in R 
                from origin o to destination d selected at time t in T
        """

        return {(o, d, t, i): int(self.h[o, d, t, i].x) for o, d in self.OD for t in self.T for i in range(1, self.K + 1)}
    
    # Get w decision variables
    def get_w(self):
        """Get the decision variables values for travel time on arc a in A at time t in T.

        Returns
        -------
        w : dict
                Dictionary of continuous variables for travel time on arc a in A at time t in T
        """

        return {(a, t): self.w[*a, t].x for a in self.A for t in self.T}
    
    # Get v decision variables
    def get_v(self):
        """Get the decision variables values for travel time from origin o to destination d at time t in T.

        Returns
        -------
        v : dict
                Dictionary of continuous variables for travel time from origin o to destination d at time t in T
        """

        return {(o, d, t): self.v[o, d, t].x for o, d in self.OD for t in self.T}    

    # Get decision variables values from set of starting times S for jobs
    def get_vars_from_times(self, S=None):
        """Get the decision variables values from a set S of starting times
        for each job in J.

        Parameters
        ----------
        S : dict, optional
                Dictionary of starting times for jobs in J, by default None
                If no input S is provided the model will check if one is stored
                in the S attribute.

        Returns
        -------
        y : dict
                Dictionary of binary variables for job j in J started at time t in T
        x : dict
                Dictionary of binary variables for arc a in A available at time t in T
        h : dict
                Dictionary of binary variables for route i in R from origin o to destination d selected at time t in T
        w : dict
                Dictionary of continuous variables for travel time on arc a in A at time t in T
        v : dict
                Dictionary of continuous variables for travel time from origin o to destination d at time t in T
        """

        # Initialize decision variables ``as if no job was started''
        y = {(j, t): 0 for j in self.J for t in self.T}
        x = {(a, t): 1 for a in self.A for t in self.T}
        h = {
            (o, d, t, i): 0
            for o, d in self.OD
            for t in self.T
            for i in range(1, self.K + 1)
        }
        w = {(a, t): self.omega_e[a] for a in self.A for t in self.T}
        v = {(o, d, t): 0 for o, d in self.OD for t in self.T}

        # Set job starting times y
        for j, t in S.items():
            y[j, t] = 1

        # Set arc availability x
        for a in self.A:
            for t in self.T:
                if any(S[j] <= t < S[j] + self.pi[j] for j in self.Ja[a]):
                    x[a, t] = 0

        # Set the arcs travel times w (constraint 4)
        for a in self.A:
            for t in self.T:
                w[a, t] = x[a, t] * self.omega_e[a] + (1 - x[a, t]) * self.omega_j[a]

        # Set route selection h and od pairs travel times v
        for t in self.T:
            for o, d in self.OD:
                # Computes times for each route by summing arcs travel times
                routes_times = [
                    sum(w[a, t] for a in self.R[(o, d)][i - 1])
                    for i in range(1, self.K + 1)
                ]

                # v[o, d, t] is the quickest route travel time
                v[o, d, t] = min(routes_times)

                # Select the route with the minimum travel time
                i = routes_times.index(min(routes_times)) + 1

                # Set the route selection h
                h[o, d, t, i] = 1  # (others already set to 0 in initialization)

        return y, x, h, w, v
    
    # Get set of starting times for jobs in J from decision variables values
    def get_times_from_vars(self, y=None):
        """Get the set of starting times S for jobs in J from the decision
        variables values.

        Parameters
        ----------
        y : dict, optional
                Dictionary of binary variables for job j in J started at time t in T, by default None
                If no input y is provided the model will check if one
                is stored in the y attribute.

        Returns
        -------
        S : dict
                Dictionary of starting times for jobs in J
        """
        
        if y is None:
            y = self.get_y()

        # Initialize the set of starting times for jobs in J
        S = {j: 0 for j in self.J}

        # Get the starting times for jobs in J
        for j in self.J:
            for t in self.T:
                if y[j, t] == 1:
                    S[j] = t

        return S

    # Get the value of the objective function
    def get_objective_value(self, v):
        """Evaluate the objective function for the railway scheduling problem.

        Parameters
        ----------
        v : dict
                Dictionary of continuous variables for travel time from origin o to destination d at time t in T

        Returns
        -------
        objective : float
                The value of the objective function for the railway scheduling problem
        """

        # Evaluate the objective function
        objective_value = sum(
            sum(self.phi[o, d, t] * (v[o, d, t] - self.Omega[o, d]) for t in self.T)
            for o, d in self.OD
        )
        return objective_value

    # Optimize -----------------------------------------------------------------

    # Optimize
    def optimize(self):
        """Optimize the railway scheduling problem."""
        
        # Solve the problem with Gurobi
        self.model.optimize()
        
        # Update starting times
        self.S = self.get_times_from_vars()

    # Check feasibility of a given solution
    def check_feasibility(self, y, x, h, w, v):
        """Check the feasibility of a given solution for the railway scheduling problem.

        Parameters
        ----------
        y : dict, optional
                Dictionary of binary variables for job j in J started at time t in T, by default None
        x : dict, optional
                Dictionary of binary variables for arc a in A available at time t in T, by default None
        h : dict, optional
                Dictionary of binary variables for route i in R from origin o to destination d selected at time t in T, by default None
        w : dict, optional
                Dictionary of continuous variables for travel time on arc a in A at time t in T, by default None
        v : dict
                Dictionary of continuous variables for travel time from origin o to destination d at time t in T

        Returns
        -------
        bool
                True if the solution is feasible, False otherwise
        """

        # Check constraints one by one
        feasible = [False] * 12

        # Job started once and completed within time horizon (2)
        feasible[0] = all(
            sum(y[j, t] for t in range(1, len(self.T) - self.pi[j] + 1)) == 1
            for j in self.J
        )

        # Availability of arc a at time t (3)
        feasible[1] = all(
            x[a, t]
            + sum(
                y[j, tp]
                for tp in range(max(1, t - self.pi[j] + 1), min(t, self.Tend) + 1)
            )
            <= 1
            for a in self.A
            for t in self.T
            for j in self.Ja[a]
        )

        # Increased travel time for service replacement (4)
        feasible[2] = all(
            w[a, t] == x[a, t] * self.omega_e[a] + (1 - x[a, t]) * self.omega_j[a]
            for a in self.A
            for t in self.T
        )

        # Ensure correct arc travel times for free variables (5)
        feasible[3] = all(
            sum(x[a, t] for t in self.T)
            == self.Tend - sum(self.pi[j] for j in self.Ja[a])
            for a in self.A
        )

        # Arcs that cannt be unavailable simultaneously (6)
        feasible[4] = all(
            sum(1 - x[a, t] for a in c) <= 1 for t in self.T for c in self.C
        )

        # Non overlapping jobs on same arc (7)
        feasible[5] = all(
            sum(
                sum(
                    y[j, tp]
                    for tp in range(max(1, t - self.pi[j] - self.tau[a] + 1), t + 1)
                )
                for j in self.Ja[a]
            )
            <= 1
            for a in self.A
            for t in self.T
        )

        # Each track segment in an event request has limited capacity (8)
        feasible[6] = all(
            sum(
                sum(
                    sum(
                        h[o, d, t, i] * self.beta[o, d, t] * self.phi[o, d, t]
                        for i in range(1, self.K + 1)
                        if a in self.R[(o, d)][i - 1]
                    )
                    for o, d in self.OD
                )
                for a in self.E[t][s]
            )
            <= sum(self.Lambd[a, t] for a in self.E[t][s])
            + (self.M * sum(x[a, t] for a in self.E[t][s]))
            for t in self.T
            for s in self.E[t]
        )

        # Passeger flow from o to d is served by one of predefined routes (9)
        feasible[7] = all(
            sum(h[o, d, t, i] for i in range(1, self.K + 1)) == 1
            for t in self.T
            for o, d in self.OD
        )

        # Lower bound for travel time from o to d (10)
        feasible[8] = all(
            v[o, d, t]
            >= sum(w[a, t] for a in self.R[(o, d)][i - 1])
            - self.M * (1 - h[o, d, t, i])
            for t in self.T
            for o, d in self.OD
            for i in range(1, self.K + 1)
        )

        # Upper bound for travel time from o to d (11)
        feasible[9] = all(
            v[o, d, t] <= sum(w[a, t] for a in self.R[(o, d)][i - 1])
            for t in self.T
            for o, d in self.OD
            for i in range(1, self.K + 1)
        )

        # Arcs never included in any job are always available (12)
        feasible[10] = all(
            x[a, t] == 1
            for a in self.A
            for t in self.T
            if not any(a in self.Aj[j] for j in self.J)
        )

        # Travel times for arcs never included in any job are equal to omega_e (13)
        feasible[11] = all(
            w[a, t] == self.omega_e[a]
            for a in self.A
            for t in self.T
            if not any(a in self.Aj[j] for j in self.J)
        )

        # Return True if all constraints are satisfied
        if all(feasible):
            return True
        else:
            for i, f in enumerate(feasible):
                if not f:
                    print(f"Constraint ({i+2}) is not satisfied.")
            return False

    # Simulated annealing (SA) optimization
    def simulated_annealing(self, T=1000, c=0.99, L=1, max_iter=1000):
        """Simulated annealing algorithm to find an initial good
        solution for the railway scheduling problem.

        Parameters
        ----------
        T : float, optional
                Initial temperature, by default 1000
        c : float, optional
                Cooling rate, by default 0.99
        L : int, optional
                Number of iterations at each temperature, by default 1
        max_iter : int, optional
                Maximum number of iterations, by default 1000

        Returns
        -------
        S : dict
                Dictionary of starting times for jobs in J obtained by the SA
                algorithm
        """

        # Initialize solution and iteration counter
        S = self.generate_initial_solution()
        iter = 0

        # SA algorithm
        while (T > 1e-6) and (iter < max_iter):
            for _ in range(L):
                # Generate a new solution
                S_new = self.generate_neighbor_solution(S)

                # Get the solutions' variables
                _, _, _, _, v = self.get_vars_from_times(S)
                _, _, _, _, v_new = self.get_vars_from_times(S_new)

                # Compute the objective function values
                f = self.get_objective_value(v)
                f_new = self.get_objective_value(v_new)

                # Accept solution if it's better or with a certain probability
                if f_new <= f:
                    S = S_new
                else:
                    p = np.exp((f - f_new) / T)
                    if np.random.rand() < p:
                        S = S_new

            # Cool down the temperature
            T *= c

            # Increment the iteration counter
            iter += 1

        # Save and return the best solution found
        self.S = S
        return S

    # Static methods -----------------------------------------------------------

    # Dijkstra's algorithm
    @staticmethod
    def dijkstra(graph, source):
        """Dijkstra's algorithm to find the shortest path in a graph.

        Parameters
        ----------
        graph : dict
                Dictionary representation of the graph
        source : int
                Source node

        Returns
        -------
        dist : dict
                Dictionary of distances from the source node to each node in the graph
        prev : dict
                Dictionary of predecessors for each node in the shortest path
        """

        # Initialize distances
        dist = {node: float("inf") for node in graph}
        dist[source] = 0
        # Initialize predecessors
        prev = {node: None for node in graph}
        # Initialize queue
        Q = deque(graph)
        while Q:
            u = min(Q, key=lambda node: dist[node])
            Q.remove(u)
            for v in graph[u]:
                alt = dist[u] + graph[u][v]
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
        return dist, prev

    # Return set of arcs of shortest path from start to end
    @staticmethod
    def shortest_arcs(predecessors, start, end):
        """Return set of arcs of shortest path from start to end.

        Parameters
        ----------
        predecessors : dict
                Dictionary of predecessors for each node in the shortest path
        start : int
                Source node	(origin	)
        end : int
                Sink node (destination)

        Returns
        -------
        path : list
                List of arcs of shortest path from start to end
        """

        path = []
        node = end
        while node != start:
            path.append((predecessors[node], node))
            node = predecessors[node]
        path.reverse()
        return path

    # Return list of nodes of shortest path from start to end
    @staticmethod
    def shortest_nodes(predecessors, start, end):
        """Return list of nodes of shortest path from start to end

        Parameters
        ----------
        predecessors : dict
                Dictionary of predecessors for each node in the shortest path
        start : int
                Source node (origin)
        end : int
                Sink node (destination)

        Returns
        -------
        path : list
                List of nodes of shortest path from start to end
        """

        path = []
        node = end
        while node is not None:
            path.append(node)
            node = predecessors[node]
        path.reverse()
        return path

    # Convert a list of nodes to a suitable list of arcs in A
    @staticmethod
    def nodes_to_arcs(nodes):
        """Convert a list of nodes to a suitable list of arcs in A

        Parameters
        ----------
        nodes : list
                List of nodes

        Returns
        -------
        path : list
                List of arcs in A
        """

        path = []
        for i in range(len(nodes) - 1):
            if nodes[i] < nodes[i + 1]:
                path.append((nodes[i], nodes[i + 1]))
            else:
                path.append((nodes[i + 1], nodes[i]))
        return path

    # Convert a list of arcs to a suitable list of nodes in N
    @staticmethod
    def arcs_to_nodes(arcs):
        """Convert a list of arcs to a suitable list of nodes in N

        Parameters
        ----------
        arcs : list
                List of arcs

        Returns
        -------
        nodes : list
                List of nodes in N

        Raises
        ------
        ValueError
                If the arcs in the input list are not connected
        """

        nodes = []
        if not arcs:
            return nodes

        # If it's just one arc, return the list of it
        if len(arcs) == 1:
            return list(arcs[0])

        # Start with the first arc in the correct order
        if arcs[0][1] == arcs[1][0] or arcs[0][1] == arcs[1][1]:
            nodes.append(arcs[0][0])
            nodes.append(arcs[0][1])
        elif arcs[0][0] == arcs[1][0] or arcs[0][0] == arcs[1][1]:
            nodes.append(arcs[0][1])
            nodes.append(arcs[0][0])
        else:
            raise ValueError("Arcs in input list are not connected")

        # Add the remaining arcs
        for i, j in arcs[1:]:
            if nodes[-1] == i:
                nodes.append(j)
            else:
                nodes.append(i)

        return nodes

    # Yen's K-shortest paths algorithm
    def YenKSP(self, graph, source, sink, K):
        """Yen's K-shortest paths algorithm to find the K-shortest paths in a graph.

        Parameters
        ----------
        graph : dict
                Dictionary representation of the graph
        source : int
                Source node (origin)
        sink : int
                Sink node (destination)
        K : int
                Number of shortest paths to find

        Returns
        -------
        A : list
                List of K-shortest paths from the source to the sink
        """

        # Initialise lists
        A = []  # list of k-shortest paths
        B = []  # list of potential k-th shortest paths

        # Determine the shortest path from the source to the sink
        dist, prev = self.dijkstra(graph, source)
        if prev[sink] == None:
            return A  # no shortest path found
        A.append((dist[sink], self.shortest_nodes(prev, source, sink)))

        for k in range(1, K):
            for i in range(len(A[-1][1]) - 1):
                spur_node = A[-1][1][
                    i
                ]  # the i-th node in the previously found k-shortest path
                root_path = A[-1][1][: i + 1]  # nodes from source to spur_node

                # Remove arcs that are part of the previous shortest paths and
                # which share the same root path
                removed_arcs = []
                for path in A:
                    if len(path[1]) > i and path[1][: i + 1] == root_path:
                        u, v = path[1][i], path[1][i + 1]
                        if v in graph[u]:
                            removed_arcs.append((u, v, graph[u][v]))
                            del graph[u][v]

                # Calculate the new spur path from the spur node to the sink
                dist, prev = self.dijkstra(graph, spur_node)
                if prev[sink] is not None:
                    spur_path = self.shortest_nodes(prev, spur_node, sink)
                    total_path = root_path[:-1] + spur_path

                    # Check for repeated nodes in the total path
                    # (we shall not have repeated nodes in a path)
                    if len(set(total_path)) == len(total_path):
                        total_cost = sum(
                            graph[total_path[i]][total_path[i + 1]]
                            for i in range(len(total_path) - 1)
                        )
                        B.append((total_cost, total_path))

                # Add back the removed arcs
                for u, v, cost in removed_arcs:
                    graph[u][v] = cost

            # Handle no spur path found case
            if not B:
                break

            # Sort potential paths by cost and
            # add the lowest cost path to the k shortest paths
            B.sort()
            A.append(B.pop(0))  # pop to remove it from B

        # Clean A and convert its elements in lists of arcs
        A = [self.nodes_to_arcs(path) for _, path in A]

        return A

    # Generators ---------------------------------------------------------------

    # Random generator method: coords (euclidean coordinates in unit circle)
    def __generate_coords(self):
        for _ in self.N:
            theta = random.uniform(0, 2 * np.pi)
            r = np.sqrt(random.uniform(0, 1))
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            self.coords.append((x, y))

    # Random generator method: pi (processing times)
    def __generate_pi(self, min_time=1, max_time=None):
        if max_time is None:
            max_time = self.Tend
        self.pi = {j: random.randint(min_time, max_time) for j in self.J}

    # Random generator method: Aj (arcs subject to jobs)
    def __generate_Aj(self, min_length=1, max_length=None):
        if max_length is None:
            max_length = self.n - 1
        self.Aj = {}
        for j in self.J:
            start_station = random.choice(self.N)  # start station
            length = random.randint(min_length, max_length)  # length of the path
            path = []
            current_station = start_station
            while len(path) < length:
                # Get all possible current station connections
                current_arcs = [a for a in self.A if current_station in a]
                # Filter out the arcs already in the path
                current_arcs = [a for a in current_arcs if a not in path]
                # If there are no more possible connections, break
                if not current_arcs:
                    break
                # Choose a random arc
                next_arc = random.choice(current_arcs)
                # Add the arc to the path
                path.append(next_arc)
                # Update the current station
                current_station = (
                    next_arc[1] if next_arc[0] == current_station else next_arc[0]
                )

            self.Aj[j] = path  # add the path to the set of arcs for job j

        # Update set of maintainance jobs on arcs
        self.__set_Ja()

    # Random generator method: tau (minimum maintainance time intervals)
    def __generate_tau(self, min_interval=0, max_interval=0):
        self.tau = {a: random.randint(min_interval, max_interval) for a in self.A}

    # Random generator method: phi (passenger demand)
    def __generate_phi(self, min_demand=0, max_demand=1):
        assert 0 <= min_demand <= 1, "min_demand must be between 0 and 1"
        assert 0 <= max_demand <= 1, "max_demand must be between 0 and 1"
        assert min_demand < max_demand, "min_demand must be less than max_demand"
        min_demand_int = int(min_demand * self.passengers)
        max_demand_int = int(max_demand * self.passengers)
        self.phi = {
            (o, d, t): random.randint(min_demand_int, max_demand_int)
            for o, d in self.OD
            for t in self.T
        }

        # Update M upper bound
        self.__set_M()

    # Random generator method: beta (share of daily passenger demand)
    def __generate_beta(self, min_share=0, max_share=1):
        assert 0 <= min_share <= 1, "min_share must be between 0 and 1"
        assert 0 <= max_share <= 1, "max_share must be between 0 and 1"
        assert min_share < max_share, "min_share must be less than max_share"
        self.beta = {
            (o, d, t): random.uniform(min_share, max_share)
            for o, d in self.OD
            for t in self.T
        }

    # Random generator method: Lambd (limited capacity of alternative services)
    def __generate_Lambd(self, min_capacity=0, max_capacity=1):
        assert 0 <= min_capacity <= 1, "min_capacity must be between 0 and 1"
        assert 0 <= max_capacity <= 1, "max_capacity must be between 0 and 1"
        assert (
            min_capacity < max_capacity
        ), "min_capacity must be less than max_capacity"
        min_capacity_int = int(min_capacity * self.passengers / self.n)
        max_capacity_int = int(max_capacity * self.passengers / self.n)
        self.Lambd = {
            (a, t): random.randint(min_capacity_int, max_capacity_int)
            for a in self.A
            for t in self.T
        }

    # Random generator method: C (set of arcs that cannot be unavailable simultaneously)
    # TODO: def __generate_C(self): ... (maybe taking in account Aj or Ja sets to avoid infeasibility)
    # NOTE: given how the constriant (6) is written, C should be a list
    # of lists of arcs that cannot be unavailable simultaneously:
    #
    # e.g. C = [ [(1, 2), (2, 3)],
    # 			 [(3, 4), (4, 5), (5, 6)] ]
    # XXX: when creating C ensure that no job needs a combination of arcs
    # that are in the same list of C otherwise the problem is infeasible...

    # Random generator method: E (set of event tracks at each time t)
    def __generate_E(self, n_max_events=0, min_length=1, max_length=None):
        if max_length is None:
            max_length = self.n - 1
        self.E = {}
        for t in self.T:
            E_tmp = {}
            n_events = random.randint(0, n_max_events)
            for _ in range(n_events):
                start_station = random.choice(self.N)  # start station
                length = random.randint(min_length, max_length)  # length of the path

                path = []
                current_station = start_station
                while len(path) < length:
                    # Get all possible current station connections
                    current_arcs = [a for a in self.A if current_station in a]
                    # Filter out the arcs already in the path
                    current_arcs = [a for a in current_arcs if a not in path]
                    # If there are no more possible connections, break
                    if not current_arcs:
                        break
                    # Choose a random arc
                    next_arc = random.choice(current_arcs)
                    # Add the arc to the path
                    path.append(next_arc)
                    # Update the current station and the next station
                    current_station = (
                        next_arc[1] if next_arc[0] == current_station else next_arc[0]
                    )

                final_station = current_station
                s = (start_station, final_station)
                E_tmp[s] = path  # add the path to the set of arcs for job j

            self.E[t] = E_tmp

    # Random generator method: R (set of routes from o to d)
    def __generate_R(self):
        self.R = {}
        for o, d in self.OD:
            self.R[(o, d)] = self.YenKSP(graph=self.graph, source=o, sink=d, K=self.K)

    # Generate problem parameters
    def generate(
        self,
        job_min_time=1,
        job_max_time=None,
        job_min_length=1,
        job_max_length=None,
        pause_min_time=0,
        pause_max_time=0,
        min_demand=0,
        max_demand=None,
        min_share=0,
        max_share=1,
        min_capacity=0,
        max_capacity=1,
        n_max_events=0,
        event_min_length=1,
        event_max_length=None,
    ):
        """Generate problem parameters for the railway scheduling problem.

        Generate random values for the problem parameters based on the provided constraints.
        In particular the method generates values for:
        - pi (processing times)
        - Aj (arcs subject to jobs)
        - tau (minimum maintainance time intervals)
        - phi (passenger demand)
        - beta (share of daily passenger demand)
        - Lambd (limited capacity of alternative services)
        - E (set of event tracks at each time t)
        - R (set of routes from o to d)

        Parameters
        ----------
        job_min_time : int, optional
                        Minimum job processing time, by default 1
        job_max_time : int, optional
                        Maximum job processing time, by default None
        job_min_length : int, optional
                        Minimum number of arcs subject to any job, by default 1
        job_max_length : int, optional
                        Maximum number of arcs subject to any job, by default None
        pause_min_time : int, optional
                        Minimum time interval between jobs on same arc, by default 0
        pause_max_time : int, optional
                        Maimum time interval between jobs on same arc, by default 0
        min_demand : float, optional
                        Minimum passenger % demand, by default 0
        max_demand : float, optional
                        Maximum passenger % demand, by default 1
        min_share : float, optional
                        Minimum share of daily passenger demand, by default 0
        max_share : float, optional
                        Maximum share of daily passenger demand, by default 1
        min_capacity : float, optional
                        Minimum capacity % per arc of alternative services, by default 0
        max_capacity : float, optional
                        Maximum capacity % per arc of alternative services, by default 1
        n_max_events : int, optional
                        Maximum number of events at each time t, by default 0
        event_min_length : int, optional
                        Minimum number of arcs in an event, by default 1
        event_max_length : int, optional
                        Maximum number of arcs in an event, by default None

        Notes
        -----
        - If `job_max_time` is not provided, it is set to `Tend`
        - If `job_max_length` is not provided, it is set to `n - 1`
        - If `event_max_length` is not provided, it is set to `n - 1`
        """

        self.__generate_pi(job_min_time, job_max_time)
        self.__generate_Aj(job_min_length, job_max_length)
        self.__generate_tau(pause_min_time, pause_max_time)
        self.__generate_phi(min_demand, max_demand)
        self.__generate_beta(min_share, max_share)
        self.__generate_Lambd(min_capacity, max_capacity)
        self.__generate_E(n_max_events, event_min_length, event_max_length)
        self.__generate_R()
        print(
            "Problem generated successfully. Remember to set constraints and objective (again)."
        )

    # Generate an initial feasible solution
    def generate_initial_solution(self):
        """Generate an initial feasible solution for the railway scheduling problem."""

        # Initialize the set of starting times for jobs and free times for the arcs
        S = {j: 1 for j in self.J}
        when_free = {a: 1 for a in self.A}

        # Build a list of jobs sorted by how many arcs they have in common with other jobs
        sorted_jobs = sorted(self.J, key=lambda j: len(self.Aj[j]), reverse=True)

        # Schedule jobs from the one with most arcs in common to the one with least
        for j in sorted_jobs:
            # Find the first available time for the job
            start_time = 1
            for a in self.Aj[j]:
                if when_free[a] > start_time:
                    start_time = when_free[a]

            # Schedule the job
            S[j] = start_time

            # Update the free times for the arcs
            for a in self.Aj[j]:
                when_free[a] = start_time + self.pi[j] + self.tau[a]

        # Get the decision variables values from the set of starting times S
        y, x, h, w, v = self.get_vars_from_times(S)

        # Check the feasibility of the solution and return
        if self.check_feasibility(y, x, h, w, v):
            return S
        else:
            raise ValueError("Initial solution is not feasible")

    # Generate a neighbor solution to a given one
    def generate_neighbor_solution(self, S):
        """Generate a neighbor solution N(S) to a given one S
        for the railway scheduling problem."""

        feasible = False
        max_tries = 10
        while not feasible and max_tries > 0:

            # Randomly chose an arc and a job on it
            a = random.choice([aj for aj, ja in self.Ja.items() if ja])
            j = random.choice(self.Ja[a])

            # Copy current solution
            Snew = S.copy()

            # Check busy and free times in the selected arc
            busy_times = [
                t
                for ja in self.Ja[a]
                if ja != j
                for t in range(S[ja] - self.tau[a], S[ja] + self.pi[ja] + self.tau[a])
            ]
            free_times = [
                t for t in range(1, self.Tend - self.pi[j] + 1) if t not in busy_times
            ]

            # If there are free times, randomly choose one
            if free_times:
                Snew[j] = random.choice(free_times)
                y, x, h, w, v = self.get_vars_from_times(Snew)

                # Return Snew only if different from S and feasible
                if Snew != S and self.check_feasibility(y, x, h, w, v):
                    return Snew

            # Update max tries
            max_tries -= 1

        # If no feasible neighbor solution was found, return None
        return None

    # Save problem parameters to a json file
    def save(self, filename):
        """Save the problem parameters to a json file.

        Parameters
        ----------
        filename : str
                The name (or path including the name) of the json file to save the problem parameters to.
        """

        with open(filename, "w") as f:
            json.dump(
                {
                    "stations": self.stations,
                    "periods": self.periods,
                    "jobs": self.jobs,
                    "passengers": self.passengers,
                    "routes": self.routes,
                    "coords": self.coords,
                    "pi": {str(k): v for k, v in self.pi.items()},
                    "Aj": {str(k): v for k, v in self.Aj.items()},
                    "C": self.C,
                    "tau": {str(k): v for k, v in self.tau.items()},
                    "phi": {str(k): v for k, v in self.phi.items()},
                    "beta": {str(k): v for k, v in self.beta.items()},
                    "Lambd": {str(k): v for k, v in self.Lambd.items()},
                    "E": {
                        str(k): {str(s): [a for a in v] for s, v in d.items()}
                        for k, d in self.E.items()
                    },
                    "R": {str(k): v for k, v in self.R.items()},
                },
                f,
            )
        print(f"Problem parameters saved successfully to {filename}")

    # TODO: PLOTTING: Add method to display the results of the optimization / solutions 
    # or other stuff like arcs vs jobs, adjacency matrix, histogram counts of jobs on arcs...
