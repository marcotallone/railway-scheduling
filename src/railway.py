# Imports
import numpy as np
import random
import gurobipy as gb
from gurobipy import Model, quicksum
from collections import deque


# Railway Class ----------------------------------------------------------------
class Railway():
	
	# Constructor
	def __init__(
		self, 
	 	n, 
	  	periods, 
	   	jobs, 
		passengers,
		K, 
		pi = {},
		Aj = {},
		C = [],
		tau = {},
		phi = {},
		beta = {},
		Lambd = {},
		E = {},
		R = {}
	):
		
		# Nodes (Stations)
		self.n = n
		self.N = range(n)
		self.coords = []
		for _ in self.N:
			theta = random.uniform(0, 2 * np.pi)
			r = np.sqrt(random.uniform(0, 1))
			x = r * np.cos(theta)
			y = r * np.sin(theta)
			self.coords.append((x, y))
   
		# Arcs (Connections)
		self.A = [(i, j) for i in self.N for j in self.N if i < j]
  
		# Travel times
		delay_factor = 1.35
		self.omega_e = {(i, j): np.sqrt(
	  		(self.coords[i][0] - self.coords[j][0])**2 
			+ (self.coords[i][1] - self.coords[j][1])**2
		) for i, j in self.A}
		self.omega_j = {a: delay_factor * self.omega_e[a] for a in self.A}
  
		# Graph
		self.graph = {node: {} for node in self.N}
		for node in self.N:
			for i, j in self.A:
				if i == node:
					self.graph[node][j] = self.omega_e[(i, j)]
				elif j == node:
					self.graph[node][i] = self.omega_e[(i, j)]
	 
		# Omega (average origin - destination travel times)
		self.Omega = {(o, d): self.dijkstra(self.graph, o)[0][d] for o in self.N for d in self.N if o != d}

		# Time Periods
		self.T = range(periods)
  
		# Jobs and processing times
		self.J = range(jobs)
		self.pi = pi
		self.Aj = Aj
		self.Ja = {a: [j for j in self.J if a in self.Aj[j]] for a in self.A}
		self.C = C
		self.tau = tau
		
		# Passengers capacity and demand
		self.passengers = passengers
		self.phi = phi
		self.beta = beta
		self.Lambd = Lambd
		sum_phi = sum(self.phi[(o, d, t)] for o in self.N for d in self.N for t in self.T if o != d)
		self.M = sum_phi + (100 * self.passengers) # unlimited capacity upper bound
  
		# Events and alternative routes
		self.E = E
		self.K = K
		self.R = R
  
		# Model
		self.model = Model()
		self.model.ModelSense = gb.GRB.MINIMIZE
  
		# Decision Variables
		self.y = self.model.addVars(self.J, self.T, vtype=gb.GRB.BINARY, name='y')
		self.x = self.model.addVars(self.A, self.T, vtype=gb.GRB.BINARY, name='x')
		self.h = self.model.addVars(self.N, self.N, self.T, range(self.K), vtype=gb.GRB.BINARY, name='h')
		self.w = self.model.addVars(self.A, self.T, lb=0, vtype=gb.GRB.CONTINUOUS, name='w')
		self.v = self.model.addVars(self.N, self.N, self.T, lb=0, vtype=gb.GRB.CONTINUOUS, name='v')
  
		# Objective Function
		self.model.setObjective(
			quicksum(
				quicksum(
					self.phi[o, d, t] * (self.v[o, d, t] - self.Omega[o, d]) for t in self.T
				) for o in self.N for d in self.N if o != d
			)
		)
  
		# Constraints
		self.__set_constraints()
  

	# Methods --------------------------------------
 
	# Dijkstra's algorithm
	@staticmethod
	def dijkstra(graph, source):
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
		path = []
		for i in range(len(nodes)-1):
			if nodes[i] < nodes[i+1]:
				path.append((nodes[i], nodes[i+1]))
			else:
				path.append((nodes[i+1], nodes[i]))
		return path

	# Convert a list of arcs to a suitable list of nodes in N
	@staticmethod
	def arcs_to_nodes(arcs):
		nodes = []
		if not arcs: return nodes

		# If it's just one arc, return the list of it
		if len(arcs) == 1: return list(arcs[0])

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
	@staticmethod
	def YenKSP(self, graph, source, sink, K):
		
		# Initialise lists
		A = [] # list of k-shortest paths
		B = [] # list of potential k-th shortest paths

		# Determine the shortest path from the source to the sink
		dist, prev = self.dijkstra(graph, source)
		if prev[sink] == None: return A # no shortest path found
		A.append((dist[sink], self.shortest_nodes(prev, source, sink)))

		for k in range(1,K):
			for i in range(len(A[-1][1]) - 1):
				spur_node = A[-1][1][i] # the i-th node in the previously found k-shortest path
				root_path = A[-1][1][:i+1] # nodes from source to spur_node

				# Remove arcs that are part of the previous shortest paths and
				# which share the same root path
				removed_arcs = []
				for path in A:
					if len(path[1]) > i and path[1][:i+1] == root_path:
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
						total_cost = sum(graph[total_path[i]][total_path[i + 1]] for i in range(len(total_path) - 1))
						B.append((total_cost, total_path))

				# Add back the removed arcs
				for u, v, cost in removed_arcs:	graph[u][v] = cost
			
			# Handle no spur path found case
			if not B: break

			# Sort potential paths by cost and 
			# add the lowest cost path to the k shortest paths
			B.sort()
			A.append(B.pop(0)) # pop to remove it from B

		# Clean A and convert its elements in lists of arcs
		A = [self.nodes_to_arcs(path) for _, path in A]

		return A

	# Constraints definition
	def __set_constraints(self):
	 
		# Minimize total passenger delays (1)
		self.model.setObjective(
			quicksum(
				quicksum(
					self.phi[o, d, t] * (self.v[o, d, t] - self.Omega[o, d]) for t in self.T
				) for o in self.N for d in self.N if o != d
			)
		)
  
		# Job started and completed within time horizon (2)
		for j in self.J:
			self.model.addConstr(
				quicksum(self.y[j, t] for t in range(len(self.T) - self.pi[j] + 1)) == 1
			)
   
		# Availability of arc a at time t (3)
		for a in self.A:
			for t in self.T:
				for j in self.Ja[a]:
					self.model.addConstr(
						self.x[*a, t] + quicksum(self.y[j, tp] for tp in range(max(0, t - self.pi[j] + 1), t)) <= 1
					)
	 
		# Increased travel time for service replacement (4)
		for a in self.A:
			for t in self.T:
				self.model.addConstr(
					self.w[*a, t] == self.x[*a, t] * self.omega_e[a] + (1 - self.x[*a, t]) * self.omega_j[a]
				)
	
		# Ensure correct arc travel times for free variables (5)
		for a in self.A:
			self.model.addConstr(
				quicksum(self.x[*a, t] for t in self.T) == len(self.T) - quicksum(self.pi[j] for j in self.Ja[a])
			)
   
		# Arcs that cannt be unavailable simultaneously (6)
		for t in self.T:
			for j in self.J:
				for c in self.C:
					self.model.addConstr(
						quicksum(1 - self.x[*a, t] for a in c) <= 1
					)
	 
		# Non overlapping jobs on same arc (7)
		for a in self.A:
			for t in self.T:
				self.model.addConstr(
					quicksum(
						quicksum(
							self.y[j, tp] for tp in range(max(0, t - self.pi[j] - self.tau[a]), len(self.T))
						) for j in self.Ja[a]
					) <= 1
				)
	
		# Each track segment in an event request has limited capacity (8)
		for t in self.T:
			for s in self.E[t]:
				self.model.addConstr(
					quicksum(
						quicksum(
							quicksum(
								self.h[o, d, t, i] * self.beta[o, d, t] * self.phi[o, d, t] for i in range(self.K) if a in self.R[(o, d)][i]
							) for o in self.N for d in self.N if o != d
						) for a in s
					) <= quicksum(self.Lambd[a, t] for a in s) + (self.M * quicksum(self.x[*a, t] for a in s))
				)

		# Passeger flow from o to d is served by one of predefined routes (9)
		for t in self.T:
			for o in self.N:
				for d in self.N:
					if o != d:
						self.model.addConstr(
							quicksum(
								self.h[o, d, t, i] for i in range(self.K)
							) == 1
						)
	  
		# Lower bound for travel time from o to d (10)
		for t in self.T:
			for o in self.N:
				for d in self.N:
					if o != d:
						for i in range(self.K):
							self.model.addConstr(
								self.v[o, d, t] >= quicksum(self.w[*a, t] for a in self.R[(o, d)][i]) - self.M * (1 - self.h[o, d, t, i])
							)
	   
		# Upper bound for travel time from o to d (11)
		for t in self.T:
			for o in self.N:
				for d in self.N:
					if o != d:
						for i in range(self.K):
							self.model.addConstr(
								self.v[o, d, t] <= quicksum(self.w[*a, t] for a in self.R[(o, d)][i])
							)
	   
	# Optimize
	def optimize(self):
		self.model.optimize()
  
	# Random geenrator method: pi (processing times)
	def __generate_pi(self, min_time=1, max_time=None):
		if max_time is None: max_time = (len(self.T) - 1) // len(self.J)
		return {j: random.randint(min_time, max_time) for j in self.J}

	# Update methdod: Ja (maintainance jobs on arcs)
	def __update_Ja(self):
		self.Ja = {a: [j for j in self.J if a in self.Aj[j]] for a in self.A}

	# Random generator method: Aj (arcs subject to jobs)
	def __generate_Aj(self, min_length=1, max_length=None):
		if max_length is None: max_length = self.n // 3
		self.Aj = {}
		for j in self.J:
			start_station = random.choice(self.N) # start station
			length = random.randint(min_length, max_length) # length of the path
   
			path = []
			current_station = start_station
			while len(path) < length:
				# Get all possible current station connections
				current_arcs = [a for a in self.A if current_station in a]
				# Filter out the arcs already in the path
				current_arcs = [a for a in current_arcs if a not in path]
				# If there are no more possible connections, break
				if not current_arcs: break
				# Choose a random arc
				next_arc = random.choice(current_arcs)
				# Add the arc to the path
				path.append(next_arc)
				# Update the current station
				current_station = next_arc[1] if next_arc[0] == current_station else next_arc[0]
	
			self.Aj[j] = path # add the path to the set of arcs for job j

		# Update set of maintainance jobs on arcs
		self.__update_Ja()
  
	# Random generator method: tau (minimum maintainance time intervals)
	def __generate_tau(self, min_interval=0, max_interval=0):
		return {a: random.randint(min_interval, max_interval) for a in self.A}

	# Random generator method: phi (passenger demand)
	def __generate_phi(self, min_demand=0, max_demand=None):
		if max_demand is None: max_demand = self.passengers
		return {(o, d, t): random.randint(min_demand, max_demand) for o in self.N for d in self.N for t in self.T if o != d}

	# Random generator method: beta (share of daily passenger demand)
	def __generate_beta(self, min_share=0.5, max_share=0.7):
		return {(o, d, t): random.uniform(min_share, max_share) for o in self.N for d in self.N for t in self.T if o != d}

	# Random generator method: Lambd (limited capacity of alternative services)
	def __generate_Lambd(self, min_capacity=None, max_capacity=None):
		if min_capacity is None: min_capacity = int(0.5 * self.passengers / self.n)
		if max_capacity is None: max_capacity = int(0.7 * self.passengers / self.n)
		return {(a, t): random.randint(min_capacity, max_capacity) for a in self.A for t in self.T}

	# Random generator method: E (set of event tracks at each time t)
	def __generate_E(self, n_max_events = 0, min_length=1, max_length=None):
		if max_length is None: max_length = (self.n - 1) // 3
		self.E = {}
		for t in self.T:
			E_tmp = []
			n_events = random.randint(0, n_max_events)
			for _ in range(n_events):
				start_station = random.choice(self.N) # start station
				length = random.randint(min_length, max_length) # length of the path

				path = []
				current_station = start_station
				while len(path) < length:
					# Get all possible current station connections
					current_arcs = [a for a in self.A if current_station in a]
					# Filter out the arcs already in the path
					current_arcs = [a for a in current_arcs if a not in path]
					# If there are no more possible connections, break
					if not current_arcs: break
					# Choose a random arc
					next_arc = random.choice(current_arcs)
					# Add the arc to the path
					path.append(next_arc)
					# Update the current station
					current_station = next_arc[1] if next_arc[0] == current_station else next_arc[0]
	 
				E_tmp.append(path) # add the path to the set of arcs for job j
	
			self.E[t] = E_tmp

	# Random generator method: R (set of routes from o to d)
	def __generate_R(self):
		self.R = {}
		for o in self.N:
			for d in self.N:
				if o != d:
					self.R[(o, d)] = self.YenKSP(self.graph, o, d, self.K)

	# Scheduling problem generator method
	def generate_problem(
			self,
			pi_min_time=1,
			pi_max_time=None,
			Aj_min_length=1,
			Aj_max_length=None,
			tau_min_interval=0,
			tau_max_interval=0,
			phi_min_demand=0,
			phi_max_demand=None,
			beta_min_share=0.5,
			beta_max_share=0.7,
			Lambd_min_capacity=None,
			Lambd_max_capacity=None,
			n_max_events=0,
			E_min_length=1,
			E_max_length=None
	):
		self.pi = self.__generate_pi(pi_min_time, pi_max_time)
		self.__generate_Aj(Aj_min_length, Aj_max_length)
		self.tau = self.__generate_tau(tau_min_interval, tau_max_interval)
		self.phi = self.__generate_phi(phi_min_demand, phi_max_demand)
		self.beta = self.__generate_beta(beta_min_share, beta_max_share)
		self.Lambd = self.__generate_Lambd(Lambd_min_capacity, Lambd_max_capacity)
		self.__generate_E(n_max_events, E_min_length, E_max_length)
		self.__generate_R()

		# Update constraints
		self.__set_constraints()
  
   
  
	# TODO: Add methods to display state of the model
 
	# TODO: Add method to display the results of the optimization / solutions 
  