# Imports
import numpy as np
import random
import gurobipy as gb
from gurobipy import Model, quicksum
from collections import deque

# Railway Class ----------------------------------------------------------------
class Railway():
    """Class modelling the railway scheduling problem.

	Attributes
	----------
	stations: int (also `n`)
		number of stations
	periods: int
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
 
	Setters
	-------
	set_constraints()
		set the constraints for the railway scheduling problem
	set_objective()
		set the objective function for the railway scheduling problem
  
	Optimize
	--------
	optimize()
		optimize and solve the railway scheduling problem
  
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
  
	"""

    # Constructor --------------------------------------------------------------
    def __init__( self, stations, periods, jobs, passengers, routes, **kwargs):

        # Attributes
        self.n = stations
        self.stations = stations
        self.periods = periods
        self.jobs = jobs
        self.passengers = passengers
        self.K = routes
        self.routes = routes

        # Sets
        self.N = range(self.n)
        self.A = [(i, j) for i in self.N for j in self.N if i < j]
        self.T = range(self.periods)
        self.J = range(self.jobs)
        self.Aj = kwargs.get('Aj', {})
        self.__set_Ja() # set the Ja set
        self.C = kwargs.get('C', [])
        self.E = kwargs.get('E', {})
        self.R = kwargs.get('R', {})

        # Parameters
        self.pi = kwargs.get('pi', {})
        self.tau = kwargs.get('tau', {})
        self.phi = kwargs.get('phi', {})
        self.beta = kwargs.get('beta', {})
        self.Lambd = kwargs.get('Lambd', {})

        # Other attributes
        self.coords = []
        for _ in self.N:
            theta = random.uniform(0, 2 * np.pi)
            r = np.sqrt(random.uniform(0, 1))
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            self.coords.append((x, y))
        self.delay_factor = 1.35
        self.omega_e = {
			(i, j): np.sqrt(
				(self.coords[i][0] - self.coords[j][0]) ** 2 +
				(self.coords[i][1] - self.coords[j][1]) ** 2
			) for i, j in self.A
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
			(o, d): self.dijkstra(self.graph, o)[0][d]
			for o in self.N
			for d in self.N
			if o != d
		}
        self.__set_M()  # set the M upper bound
        self.model = Model()
        self.model.ModelSense = gb.GRB.MINIMIZE

        # Decision Variables
        self.y = self.model.addVars(self.J, self.T, vtype=gb.GRB.BINARY, name="y")
        self.x = self.model.addVars(self.A, self.T, vtype=gb.GRB.BINARY, name="x")
        self.h = self.model.addVars(
			self.N, self.N, self.T, range(self.K), vtype=gb.GRB.BINARY, name="h"
		)
        self.w = self.model.addVars(
			self.A, self.T, lb=0, vtype=gb.GRB.CONTINUOUS, name="w"
		)
        self.v = self.model.addVars(
			self.N, self.N, self.T, lb=0, vtype=gb.GRB.CONTINUOUS, name="v"
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
            self.M = sum(
				self.phi[(o, d, t)]
				for o in self.N
				for d in self.N
				for t in self.T
				if o != d
			) + (100 * self.passengers)

    # Set constraints
    def set_constraints(self):
        """Set the constraints for the railway scheduling problem."""

        # # Job started and completed within time horizon (2)
        # for j in self.J:
        #     self.model.addConstr(
        # 		quicksum(self.y[j, t] for t in range(len(self.T) - self.pi[j] + 1)) == 1
        # 	)

        # Availability of arc a at time t (3)
        # for a in self.A:
        #     for t in self.T:
        #         for j in self.Ja[a]:
        #             self.model.addConstr(
        # 				self.x[*a, t] + quicksum(self.y[j, tp] for tp in range(max(0, t - self.pi[j] + 1), t)) <= 1
        # 			)

        # Increased travel time for service replacement (4)
        # for a in self.A:
        #     for t in self.T:
        #         self.model.addConstr(
        # 			self.w[*a, t] == self.x[*a, t] * self.omega_e[a] + (1 - self.x[*a, t]) * self.omega_j[a]
        # 		)

        # # Ensure correct arc travel times for free variables (5)
        # for a in self.A:
        #     self.model.addConstr(
        # 		quicksum(self.x[*a, t] for t in self.T) == len(self.T) - quicksum(self.pi[j] for j in self.Ja[a])
        # 	)

        # # Arcs that cannt be unavailable simultaneously (6)
        # for t in self.T:
        #     for j in self.J:
        #         for c in self.C:
        #             self.model.addConstr(
        # 				quicksum(1 - self.x[*a, t] for a in c) <= 1
        # 			)

        # # Non overlapping jobs on same arc (7)
        # for a in self.A:
        #     for t in self.T:
        #         self.model.addConstr(
        # 			quicksum(
        # 				quicksum(
        # 					self.y[j, tp] for tp in range(max(0, t - self.pi[j] - self.tau[a]), len(self.T))
        # 				) for j in self.Ja[a]
        # 			) <= 1
        # 		)

        # # Each track segment in an event request has limited capacity (8)
        # for t in self.T:
        #     for s in self.E[t]:
        #         self.model.addConstr(
        # 			quicksum(
        # 				quicksum(
        # 					quicksum(
        # 						self.h[o, d, t, i] * self.beta[o, d, t] * self.phi[o, d, t] for i in range(self.K) if a in self.R[(o, d)][i]
        # 					) for o in self.N for d in self.N if o != d
        # 				) for a in s
        # 			) <= quicksum(self.Lambd[a, t] for a in s) + (self.M * quicksum(self.x[*a, t] for a in s))
        # 		)

        # # Passeger flow from o to d is served by one of predefined routes (9)
        # for t in self.T:
        #     for o in self.N:
        #         for d in self.N:
        #             if o != d:
        #                 self.model.addConstr(
        # 					quicksum(
        # 						self.h[o, d, t, i] for i in range(self.K)
        # 					) == 1
        # 				)

        # # Lower bound for travel time from o to d (10)
        # for t in self.T:
        #     for o in self.N:
        #         for d in self.N:
        #             if o != d:
        #                 for i in range(self.K):
        #                     self.model.addConstr(
        # 						self.v[o, d, t] >= quicksum(self.w[*a, t] for a in self.R[(o, d)][i]) - self.M * (1 - self.h[o, d, t, i])
        # 					)

        # # Upper bound for travel time from o to d (11)
        # for t in self.T:
        #     for o in self.N:
        #         for d in self.N:
        #             if o != d:
        #                 for i in range(self.K):
        #                     self.model.addConstr(
        # 						self.v[o, d, t] <= quicksum(self.w[*a, t] for a in self.R[(o, d)][i])
        # 					)



        # TODO: improve constraints efficiency when setting...

        # ---------------------------------------------------------------------- jobs loop
        for j in self.J:

            # Job started and completed within time horizon                     (2)
            self.model.addConstr(
				quicksum(self.y[j, t] for t in range(len(self.T) - self.pi[j])) == 1
			)

        # ---------------------------------------------------------------------- arcs loop
        for a in self.A:

            # Ensure correct arc travel times for free variables                 (5)
            self.model.addConstr(
                quicksum(self.x[*a, t] for t in self.T) == len(self.T) - quicksum(self.pi[j] for j in self.Ja[a])
            )

        # ---------------------------------------------------------------------- times loop
        for t in self.T:

            for a in self.A:

                for j in self.Ja[a]:

                    # Availability of arc a at time t                           (3)
                    t_start = max(0, t - self.pi[j])
                    self.model.addConstr(
                        self.x[*a, t] + quicksum(self.y[j, tp] for tp in range(t_start, t)) <= 1
                    )

                # Increased travel time for service replacement                 (4)
                self.model.addConstr(
                    self.w[*a, t] == self.x[*a, t] * self.omega_e[a] + (1 - self.x[*a, t]) * self.omega_j[a]
                )

                # Non overlapping jobs on same arc                              (7)
                self.model.addConstr(
                    quicksum(
                        quicksum(
                            self.y[j, tp] for tp in range( max(0, t - self.pi[j] - self.tau[a]), t)
                        ) for j in self.Ja[a]
                    )
                    <= 1
                )

                # Arcs never included in any job are always available           (12)
                if not any(a in self.Aj[j] for j in self.J):
                    self.model.addConstr(self.x[*a, t] == 1)
                    
                # Travel times for arcs never included in any job               (13)
                if not any(a in self.Aj[j] for j in self.J):
                    self.model.addConstr(self.w[*a, t] == self.omega_e[a])

            # for j in self.J: <-- TODO: willingly C can depend on job j os this extenal loop is needed
            # Arcs that cannot be unavailable simultaneously                    (6)
            for c in self.C:
                self.model.addConstr(
                    quicksum(1 - self.x[*a, t] for a in c) <= 1
                )

            for s in self.E[t]:

                # Each track segment in an event request has limited capacity   (8)
                self.model.addConstr(
                    quicksum(
                        quicksum(
                            quicksum(
                                self.h[o, d, t, i] * self.beta[o, d, t] * self.phi[o, d, t] for i in range(self.K) if a in self.R[(o, d)][i]
                            ) for o in self.N for d in self.N if o != d
                        ) for a in s
                    ) <= quicksum(self.Lambd[a, t] for a in s) + (self.M * quicksum(self.x[*a, t] for a in s))
                )

                for a in s:
                    for o in self.N:
                        for d in self.N:
                            if o != d:
                                # Availability of arc included in all event paths (14)
                                if all(a in self.R[(o, d)][i] for i in range(self.K)):
                                    self.model.addConstr(self.x[*a, t] == 1)

            for o in self.N:
                for d in self.N:
                    if o != d:

                        # Passenger flow o -> d served by one of the K routes   (9)
                        self.model.addConstr(
                            quicksum(self.h[o, d, t, i] for i in range(self.K)) == 1
                        )

                        for i in range(self.K):
                            
                            # Lower bound for travel time from o to d           (10)
                            self.model.addConstr(
                                self.v[o, d, t] >= quicksum(self.w[*a, t] for a in self.R[(o, d)][i]) - self.M * (1 - self.h[o, d, t, i])
                            )

                            # Upper bound for travel time from o to d           (11)
                            self.model.addConstr(
                                self.v[o, d, t] <= quicksum(self.w[*a, t] for a in self.R[(o, d)][i])
                            )
                            

    # # Set fixed values
    # def set_fixed_values(self):
    #     """Set fixed values for decision variables for reduction of solution space."""

    #     # Arcs never included in any job are always available (i.e. x = 1)
    #     self.model.addConstrs(
	# 		self.x[a, t] == 1 for a in self.A for t in self.T if not any(a in self.Aj[j] for j in self.J)
	# 	)

    #     # Travel times for arcs never included in any job are equal to the average travel time (i.e. w = omega_e)
    #     self.model.addConstrs(
	# 		self.w[a, t] == self.omega_e[a] for a in self.A for t in self.T if not any(a in self.Aj[j] for j in self.J)
	# 	)

    #     # If an arc a is included in all the possible K paths at a certain event at time t then that arc must be available (i.e. x = 1)
    #     for t in self.T:
    #         for s in self.E[t]:
    #             for a in s:
    #                 for o in self.N:
    #                     for d in self.N:
    #                         if o != d:
    #                             if all(a in self.R[(o, d)][i] for i in range(self.K)) and (self.beta[o, d, t]*self.phi[o, d, t] > self.Lambd[a, t]):
    #                                 self.model.addConstr(self.x[a, t] == 1)					

    # Set objective function
    def set_objective(self):
        """Set the objective function for the railway scheduling problem."""
        self.model.setObjective(
			quicksum(
				quicksum(
					self.phi[o, d, t] * (self.v[o, d, t] - self.Omega[o, d]) for t in self.T
				) for o in self.N for d in self.N if o != d
			)
		)

        # Optimize ------------------------------------------------------------------

    # Optimize
    def optimize(self):
        """Optimize the railway scheduling problem."""
        self.model.optimize()

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

    # Random generator method: pi (processing times)
    def __generate_pi(self, min_time=1, max_time=None):
        if max_time is None:
            max_time = (len(self.T) - 1) // len(self.J)
        # return {j: random.randint(min_time, max_time) for j in self.J}
        self.pi = {j: random.randint(min_time, max_time) for j in self.J}

    # Random generator method: Aj (arcs subject to jobs)
    def __generate_Aj(self, min_length=1, max_length=None):
        if max_length is None:
            max_length = self.n // 3
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
        # return {a: random.randint(min_interval, max_interval) for a in self.A}
        self.tau = {a: random.randint(min_interval, max_interval) for a in self.A}

    # Random generator method: phi (passenger demand)
    def __generate_phi(self, min_demand=0, max_demand=None):
        if max_demand is None:
            max_demand = self.passengers
        # return {(o, d, t): random.randint(min_demand, max_demand) for o in self.N for d in self.N for t in self.T if o != d}
        self.phi = {
            (o, d, t): random.randint(min_demand, max_demand)
            for o in self.N
            for d in self.N
            for t in self.T
            if o != d
        }
        
        # Update M upper bound
        self.__set_M()

    # Random generator method: beta (share of daily passenger demand)
    def __generate_beta(self, min_share=0.5, max_share=0.7):
        # return {(o, d, t): random.uniform(min_share, max_share) for o in self.N for d in self.N for t in self.T if o != d}
        self.beta = {
            (o, d, t): random.uniform(min_share, max_share)
            for o in self.N
            for d in self.N
            for t in self.T
            if o != d
        }

    # Random generator method: Lambd (limited capacity of alternative services)
    def __generate_Lambd(self, min_capacity=None, max_capacity=None):
        if min_capacity is None:
            min_capacity = int(0.5 * self.passengers / self.n)
        if max_capacity is None:
            max_capacity = int(0.7 * self.passengers / self.n)
        # return {(a, t): random.randint(min_capacity, max_capacity) for a in self.A for t in self.T}
        self.Lambd = {
            (a, t): random.randint(min_capacity, max_capacity)
            for a in self.A
            for t in self.T
        }

    # Random generator method: E (set of event tracks at each time t)
    def __generate_E(self, n_max_events=0, min_length=1, max_length=None):
        if max_length is None:
            max_length = (self.n - 1) // 3
        self.E = {}
        for t in self.T:
            E_tmp = []
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
                    # Update the current station
                    current_station = (
                        next_arc[1] if next_arc[0] == current_station else next_arc[0]
                    )

                E_tmp.append(path)  # add the path to the set of arcs for job j

            self.E[t] = E_tmp

    # Random generator method: R (set of routes from o to d)
    def __generate_R(self):
        self.R = {}
        for o in self.N:
            for d in self.N:
                if o != d:
                    self.R[(o, d)] = self.YenKSP(
                        graph=self.graph, source=o, sink=d, K=self.K
                    )

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
        min_share=0.5,
        max_share=0.7,
        min_capacity=None,
        max_capacity=None,
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
        min_demand : int, optional
                Minimum passenger demand, by default 0
        max_demand : int, optional
                Maximum passenger demand, by default None
        min_share : float, optional
                Minimum share of daily passenger demand, by default 0.5
        max_share : float, optional
                Maximum share of daily passenger demand, by default 0.7
        min_capacity : int, optional
                Minimum capacity of alternative services, by default None
        max_capacity : int, optional
                Maximum capacity of alternative services, by default None
        n_max_events : int, optional
                Maximum number of events at each time t, by default 0
        event_min_length : int, optional
                Minimum number of arcs in an event, by default 1
        event_max_length : int, optional
                Maximum number of arcs in an event, by default None

        Notes
        -----
        - If `job_max_time` is not provided, it is set to the number of periods divided by the number of jobs.
        - If `job_max_length` is not provided, it is set to the number of stations divided by 3.
        - If `max_demand` is not provided, it is set to the total number of passengers.
        - If `event_max_length` is not provided, it is set to the number of stations minus 1
        """

        self.__generate_pi(job_min_time, job_max_time)
        self.__generate_Aj(job_min_length, job_max_length)
        self.__generate_tau(pause_min_time, pause_max_time)
        self.__generate_phi(min_demand, max_demand)
        self.__generate_beta(min_share, max_share)
        self.__generate_Lambd(min_capacity, max_capacity)
        self.__generate_E(n_max_events, event_min_length, event_max_length)
        self.__generate_R()
        print("Problem generated successfully. Remember to set constraints and objective (again).")

    # TODO: Add methods to display state of the model

    # TODO: Add method to display the results of the optimization / solutions
