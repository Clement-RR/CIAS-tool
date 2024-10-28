import numpy as np
from scipy.stats import beta


class BreadthFirstMC:
    """
    A class to perform a breadth-first Monte Carlo simulation.

    Attributes:
         P (np.ndarray): Transition probabilities matrix.
        A (np.ndarray): Starting cost matrix.
        B (np.ndarray): Final cost matrix.
        tA (np.ndarray): Starting time matrix.
        ta (np.ndarray): Alpha parameters for beta distribution of time.
        tb (np.ndarray): Beta parameters for beta distribution of time.
        ct (float): Cost per time unit.

    Methods:
        run(node, depth, allow): Performs the simulation from a given node, to a certain depth, with the option to allow revisiting nodes.
        """

    def __init__(self, P, A, B, tA, ta, tb, ct):
        self.P = P
        self.A = A
        self.B = B
        self.tA = tA
        self.ta = ta
        self.tb = tb
        self.ct = ct

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
