"""
This module defines the Simulation class, which encapsulates the logic for running
Monte Carlo simulations to analyze the impact of various parameters on costs,
times, and total costs in a given system. It includes methods for initializing
beta parameters, running the simulation, and performing breadth-first search
Monte Carlo and expected value calculations.
"""

import numpy as np
from scipy.stats import beta
from utils import Utils


class Simulation:
    """
    A class to perform Monte Carlo simulations with breadth-first search techniques.

    Attributes:
        P, A, B, M, tA, tB, tM (numpy.ndarray): Input matrices defining probabilities and costs.
        start (int): The starting node for the simulation.
        depth (int): Maximum depth of the search.
        trials (int): Number of trials to run in the simulation.
        ct (int): Cost per time unit.
        all_sum, all_sum_sqr, cost_cum, time_cum, total_cum (numpy.ndarray): Accumulators for
        simulation results.

    Methods:
        initialize_beta_parameters(): Initializes beta distribution parameters for costs and times.
        calculate_beta_parameters(A, B, M): Calculates the parameters of the beta distribution.
        run_simulation(): Runs the Monte Carlo simulation and aggregates results.
        breadth_first_mc(node, depth, allow): Performs a Monte Carlo simulation using a
        breadth-first search approach.
        breadth_first_exp(node, depth, allow): Calculates expected costs, times, and total costs
        using a breadth-first search.
    """
    def __init__(self, inP, inA, inB, inM, intA, intB, intM, start, depth, trials, ct=560):
        self.P = inP
        self.A = inA
        self.B = inB
        self.M = inM
        self.tA = intA
        self.tB = intB
        self.tM = intM
        self.start = start
        self.depth = depth
        self.trials = trials
        self.ct = ct
        self.all_sum = 0
        self.all_sum_sqr = 0
        self.cost_cum = np.zeros_like(self.P)
        self.time_cum = np.zeros_like(self.P)
        self.total_cum = np.zeros_like(self.P)
        self.initialize_beta_parameters()

    def initialize_beta_parameters(self):
        self.mu, self.sig, self.a, self.b = self.calculate_beta_parameters(self.A, self.B, self.M)
        self.tmu, self.tsig, self.ta, self.tb = \
            self.calculate_beta_parameters(self.tA, self.tB, self.tM)

    @staticmethod
    def calculate_beta_parameters(A, B, M):
        denominator = (B - A)
        valid = ~np.isclose(denominator, 0) & ~np.isnan(denominator)
        mu = (A + 4 * M + B) / 6
        sig = (B - A) / 6
        a = np.zeros_like(mu, dtype=float)
        b = np.zeros_like(mu, dtype=float)
        a[valid] = (mu[valid] - A[valid]) / denominator[valid] * (
                (mu[valid] - A[valid]) * (B[valid] - mu[valid]) / sig[valid] ** 2 - 1)
        b[valid] = (B[valid] - mu[valid]) / denominator[valid] * (
                (mu[valid] - A[valid]) * (B[valid] - mu[valid]) / sig[valid] ** 2 - 1)
        return mu, sig, a, b

    def run_simulation(self):
        """
        Runs the Monte Carlo simulation across all trials, aggregates results,
        and prepares for analysis.
        """
        # Initialize accumulators for costs, times, and total costs
        self.cost_cum = np.zeros_like(self.P)  # Initialize invest output array for storage
        self.time_cum = np.zeros_like(self.P)  # Initialize time output array for storage
        self.total_cum = np.zeros_like(self.P)  # Initialize total cost output array for storage

        cost_cum_sqr = np.zeros_like(self.P)  # Initialize invest output array for storage
        time_cum_sqr = np.zeros_like(self.P)  # Initialize time output array for storage
        total_cum_sqr = np.zeros_like(self.P)  # Initialize total cost output array for storage

        # Initialize scalars for total sum and total sum squared of costs for all trials
        self.all_sum = 0
        self.all_sum_sqr = 0

        # Lists to store results for histogram plotting
        self.cost_cum_list = []
        self.time_cum_list = []
        self.total_cum_list = []

        for _ in range(self.trials):
            # Simulate costs, times, and totals for one trial
            cost, time, total = \
                self.breadth_first_mc(self.start, self.depth, 0)

            # Filter the results to keep only the minimum non-zero values per column
            cost = Utils.mincolumn(cost)
            time = Utils.mincolumn(time)
            total = Utils.mincolumn(total)

            # Accumulate results
            self.cost_cum += cost
            self.time_cum += time
            self.total_cum += total

            # Append sums to lists for histogram plotting
            self.cost_cum_list.append(np.sum(cost))
            self.time_cum_list.append(np.sum(time))
            self.total_cum_list.append(np.sum(total))

            # Update overall sums and squares
            self.all_sum += np.sum(total)
            self.all_sum_sqr += np.sum(total) ** 2

            # Update squared sums for variance calculation
            cost_cum_sqr += total ** 2
            time_cum_sqr += time ** 2
            total_cum_sqr += total ** 2

    def breadth_first_mc(self, node, depth, allow):
        """
        Performs a breadth-first search Monte Carlo simulation.

        Args:
            node (int): The starting node.
            depth (int): The depth of the search.
            allow (int): Flag to allow revisiting nodes.

        Returns:
            Tuple of np.arrays: costs, times, and total costs matrices.
        """
        # Initialization of variables before the loop
        nodeslist = [node]  # List of initial start nodes IDs
        probslist = [1.0] * len(nodeslist)  # Initialize transition probabilities with 1s
        visited = set()  # Keep track of visited nodes using a set for efficiency

        costs1 = np.zeros_like(self.P)  # Initialize costs accumulator
        costs2 = np.zeros_like(self.P)  # Initialize time accumulator
        costs3 = np.zeros_like(self.P)  # Initialize total cost accumulator

        for depthiter in range(depth):  # Iterate up to the specified depth
            tempnodeslist = []  # Temporary storage for next nodes
            tempprobslist = []  # Temporary storage for their probabilities

            while nodeslist:  # While there are nodes in the list
                n = nodeslist[0]  # Get and remove the first node from the list
                nodeslist = nodeslist[1:]
                visited.add(n - 1)  # Mark it as visited
                p = probslist[0]  # Get and remove its associated probability
                probslist = probslist[1:]

                v = self.P[n - 1]

                # TODO: Validate data without random generation

                for i, v_prob in enumerate(v):  # Iterate over possible next nodes
                    if np.random.rand() < v_prob:  # With probability v, proceed
                        # Calculate money and time costs based on beta distributions
                        money = self.A[n - 1, i] + (self.B[n - 1, i] - self.A[n - 1, i]) * \
                                beta.rvs(self.a[n - 1, i], self.b[n - 1, i])
                        time = self.tA[n - 1, i] + (self.tB[n - 1, i] - self.tA[n - 1, i]) * \
                                beta.rvs(self.ta[n - 1, i], self.tb[n - 1, i])

                        # Accumulate costs
                        costs1[n - 1, i] += money
                        costs2[n - 1, i] += time
                        costs3[n - 1, i] += money + self.ct * time

                        if allow or i not in visited:  # If revisiting is allowed or node unvisited
                            tempnodeslist.append(i)  # Add to the list for the next iteration
                            tempprobslist.append(p * v_prob)  # Update the transition probability

            # Prepare for the next depth iteration
            nodeslist = tempnodeslist
            probslist = tempprobslist

        return costs1, costs2, costs3

    def breadth_first_exp(self, node, depth, allow):
        """
        Performs a breadth-first search to calculate expected costs, times,
        and total costs up to a given depth, optionally allowing revisiting of nodes.

        This method models the expected impact (in terms of costs and time)
        of starting from a specific node and explores transitions up to a specified depth,
        taking into account the probability of each transition and the expected
        costs and times as defined by beta distributions.

        Parameters: - node (int): The starting node index for the search.
        - depth (int): The maximum depth of the search, determining how far
        from the starting node the search should explore.
        - allow (bool): Flag indicating whether nodes can be revisited.
        If True, nodes can be revisited in the search.
        If False, each node is visited at most once.

        Returns:
        - np.array: An array containing three elements representing the total expected costs,
        total expected time, and total expected cost (including time cost),
        accumulated over all paths explored up to the specified depth.

        Note:
        - This method assumes that all transition probabilities, costs, and times are predefined
        in the class attributes `P`, `A`, `B`, `M`, `tA`, `tB`, `tM`, and uses beta distribution
        parameters `a`, `b`, `ta`, `tb` for calculating expected values.
        """
        nodeslist = [node]  # Starting node
        visited = set()  # Track visited nodes
        costs = np.zeros(3)  # Initialize costs accumulator for money, time, and total cost
        probslist = [1.0] * len(nodeslist)  # Initialize with probability 1

        for depthiter in range(depth):
            tempnodeslist = []  # Temporary storage for nodes
            tempprobslist = []  # Temporary storage for probabilities

            while nodeslist:
                n = nodeslist.pop(0)  # Current node
                p = probslist.pop(0)  # Current probability

                # Ensure only unvisited nodes are processed unless revisiting is allowed
                if allow or n not in visited:
                    visited.add(n)  # Mark node as visited

                    for i, v in enumerate(self.P[n]):  # Iterate over possible transitions
                        # Check if transition probability is significant
                        if p * v > 0.00001:
                            # Expected money and time based on beta parameters
                            money = self.A[n, i] + (self.B[n, i] - self.A[n, i]) * (
                                    self.a[n, i] / (self.a[n, i] + self.b[n, i]))
                            time = self.tA[n, i] + (self.tB[n, i] - self.tA[n, i]) * (
                                    self.ta[n, i] / (self.ta[n, i] + self.tb[n, i]))

                            # Accumulate expected costs
                            costs[0] += money * p * v
                            costs[1] += time * p * v
                            costs[2] += (money + self.ct * time) * p * v

                            # Prepare for next level if node is unvisited or revisiting is allowed
                            if allow or i not in visited:
                                tempnodeslist.append(i)
                                tempprobslist.append(p * v)

            # Update lists for the next iteration
            nodeslist = tempnodeslist
            probslist = tempprobslist

        return costs
