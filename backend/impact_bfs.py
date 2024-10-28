import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from input_data import P, A, B, M, tA, tB, tM, start, depth, trials
from mpl_toolkits.mplot3d import Axes3D
from functools import wraps
import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print('func:%r took: %2.4f sec' % \
              (f.__name__, te - ts))
        return result

    return wrap


class ImpactBFS:
    def __init__(self, inP, inA, inB, inM, intA, intB, intM, start, depth, trials):
        self.all_sum_sqr = None
        self.all_sum = None
        self.total_cum = None
        self.time_cum = None
        self.total_cum_list = None
        self.time_cum_list = None
        self.cost_cum_list = None
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
        self.cost_cum = np.zeros_like(self.P)
        self.ct = 560  # Cost per day initialization

        denominator = (self.B - self.A)
        v_d = ~np.isclose(denominator, 0) & ~np.isnan(
            denominator)  # True where denominator is neither close to zero nor NaN

        # Calculate mu, sig, a, b for costs and times
        self.mu = (self.A + 4 * self.M + self.B) / 6
        self.sig = (self.B - self.A) / 6

        # Initialize a and b with NaNs or zeros
        self.a = np.zeros_like(self.mu, dtype=float)
        self.b = np.zeros_like(self.mu, dtype=float)

        # Only perform division where the denominator is valid
        self.a[v_d] = (self.mu[v_d] - self.A[v_d]) / denominator[v_d] * (
                    (self.mu[v_d] - self.A[v_d]) * (self.B[v_d] - self.mu[v_d]) / self.sig[v_d] ** 2 - 1)
        self.b[v_d] = (self.B[v_d] - self.mu[v_d]) / denominator[v_d] * (
                    (self.mu[v_d] - self.A[v_d]) * (self.B[v_d] - self.mu[v_d]) / self.sig[v_d] ** 2 - 1)

        t_denominator = (self.tB - self.tA)
        t_v_d = ~np.isclose(t_denominator, 0) & ~np.isnan(
            t_denominator)  # True where denominator is neither close to zero nor NaN

        self.tmu = (self.tA + 4 * self.tM + self.tB) / 6
        self.tsig = (self.tB - self.tA) / 6

        # Initialize a and b with NaNs or zeros
        self.ta = np.zeros_like(self.tmu, dtype=float)
        self.tb = np.zeros_like(self.tmu, dtype=float)

        self.ta[v_d] = (self.tmu[t_v_d] - self.tA[t_v_d]) / t_denominator[t_v_d] * (
                    (self.tmu[t_v_d] - self.tA[t_v_d]) * (self.tB[t_v_d] - self.tmu[t_v_d]) / self.tsig[t_v_d] ** 2 - 1)
        self.tb[v_d] = (self.tB[t_v_d] - self.tmu[t_v_d]) / t_denominator[t_v_d] * (
                    (self.tmu[t_v_d] - self.tA[t_v_d]) * (self.tB[t_v_d] - self.tmu[t_v_d]) / self.tsig[t_v_d] ** 2 - 1)

    @timing
    def run_simulation(self):
        """
        Runs the Monte Carlo simulation across all trials, aggregates results, and prepares for analysis.
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
            cost, time, total = self.breadth_first_mc(self.start, self.depth, 0)  # Assuming allow revisiting disabled

            # Filter the results to keep only the minimum non-zero values per column
            cost = self.mincolumn(cost)
            time = self.mincolumn(time)
            total = self.mincolumn(total)

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

        self.plot_results()

    @staticmethod
    def mincolumn(m):
        """
        Identifies the minimum non-zero value in each column of the input matrix 'm',
        setting all other values to 0, effectively filtering out non-minimum values.

        Args:
        m (np.array): Input 2D numpy array.

        Returns:
        np.array: A 2D numpy array with the same shape as 'm', where each column contains
                  only its minimum non-zero value, with all other entries set to 0.
        """
        A = np.zeros_like(m)  # Initialize A with zeros of the same shape as m

        for i in range(m.shape[1]):  # Iterate over columns
            column = m[:, i]
            non_zero = column[column != 0]  # Filter out zero values

            if non_zero.size > 0:  # Check if there are non-zero values
                min_val = np.min(non_zero)  # Find the minimum non-zero value
                min_positions = column == min_val  # Identify positions of the minimum value
                A[min_positions, i] = column[min_positions]  # Copy these minimum values to A

        return A

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
                visited.add(n-1)  # Mark it as visited
                p = probslist[0]   # Get and remove its associated probability
                probslist = probslist[1:]

                v = self.P[n-1]

                # TODO: Validate data without random generation

                for i, v_prob in enumerate(v):  # Iterate over possible next nodes
                    if np.random.rand() < v_prob:  # With probability v, proceed
                        # Calculate money and time costs based on beta distributions
                        money = self.A[n-1, i] + (self.B[n-1, i] - self.A[n-1, i]) * beta.rvs(self.a[n-1, i], self.b[n-1, i])
                        time = self.tA[n-1, i] + (self.tB[n-1, i] - self.tA[n-1, i]) * beta.rvs(self.ta[n-1, i], self.tb[n-1, i])

                        # Accumulate costs
                        costs1[n-1, i] += money
                        costs2[n-1, i] += time
                        costs3[n-1, i] += money + self.ct * time

                        if allow or i not in visited:  # If revisiting is allowed or node is unvisited
                            tempnodeslist.append(i)  # Add to the list for the next iteration
                            tempprobslist.append(p * v_prob)  # Update the transition probability

            # Prepare for the next depth iteration
            nodeslist = tempnodeslist
            probslist = tempprobslist

        return costs1, costs2, costs3

    @timing
    def breadth_first_exp(self, node, depth, allow):
        """
        Performs a breadth-first search to calculate expected costs, times, and total costs up to a given depth,
        optionally allowing revisiting of nodes.

        This method models the expected impact (in terms of costs and time) of starting from a specific node and
        explores transitions up to a specified depth, taking into account the probability of each transition and the
        expected costs and times as defined by beta distributions.

        Parameters: - node (int): The starting node index for the search.
        - depth (int): The maximum depth of the search, determining how far from the starting node the search should
          explore.
        - allow (bool): Flag indicating whether nodes can be revisited. If True, nodes can be revisited in the search.
          If False, each node is visited at most once.

        Returns:
        - np.array: An array containing three elements representing the total expected costs, total expected time, and
          total expected cost (including time cost), accumulated over all paths explored up to the specified depth.

        Note:
        - This method assumes that all transition probabilities, costs, and times are predefined in the class attributes
          `P`, `A`, `B`, `M`, `tA`, `tB`, `tM`, and uses beta distribution parameters `a`, `b`, `ta`, `tb` for
           calculating expected values.
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

    @timing
    def plot_results(self):
        """
        Plots histograms for required investments, required time, and total cost based on simulation results.

        This method uses the Rice Rule to determine the bin size for histograms and plots separate
        histograms for accumulated costs, times, and total costs from the simulation.
        """
        # Calculate bin size using the Rice Rule
        binsize = int(2 * np.cbrt(self.trials))

        # Plot histogram of required investments
        plt.figure()
        plt.hist(self.cost_cum_list, bins=binsize, color='#C0504D')
        plt.title('Prediction of required investments (n = 20,000)')
        plt.xlabel('Monetary units')
        plt.ylabel('Frequency')

        # Plot histogram of required time
        plt.figure()
        plt.hist(self.time_cum_list, bins=binsize, color='#6495ED')
        plt.title('Prediction of person-hours required for implementation (n = 20,000)')
        plt.xlabel('Person-hours')
        plt.ylabel('Frequency')

        # Plot histogram of required total cost
        plt.figure()
        plt.hist(self.total_cum_list, bins=binsize, color='#9C9C9C')
        plt.title('Prediction of total cost (n = 20,000)')
        plt.xlabel('Monetary units')
        plt.ylabel('Frequency')

        plt.show()

        self.plot_3d_bar_chart(self.cost_cum / self.trials, title='Heatmap of Expected Invest', xlabel='Affected',
                               ylabel='Initiating', zlabel='Monetary Units')

    @timing
    def print_simulation_summary(self):
        """
        Prints a summary of the Monte Carlo simulation results, including
        mean values of costs, times, and total costs, and calculates the standard deviation
        for total costs.
        """
        # Calculating mean values
        mean_cost = np.sum(self.cost_cum) / self.trials
        mean_time = np.sum(self.time_cum) / self.trials
        mean_total_cost = np.sum(self.total_cum) / self.trials

        # Calculating total mean for all trials
        total_mean = np.sum(self.all_sum) / self.trials

        # Calculating standard deviation for total costs
        # std_dev_total_cost = np.sqrt(np.sum((self.all_sum_sqr - mean_total_cost) ** 2) / self.trials)
        std_dev_total_cost = np.sqrt(self.all_sum_sqr / self.trials - (self.all_sum / self.trials) ** 2)

        # Displaying the results
        print("Monte Carlo Simulation Results:")
        print(f"Mean Cost: {mean_cost}")
        print(f"Mean Time: {mean_time}")
        print(f"Mean Total Cost: {mean_total_cost}")
        print(f"Total Mean: {total_mean}")
        print(f"Standard Deviation of Total Cost: {std_dev_total_cost}")

    @staticmethod
    @timing
    def plot_3d_bar_chart(data, title='Heatmap of Expected Invest', xlabel='Affected', ylabel='Initiating',
                          zlabel='Monetary Units'):
        """
        Plots a 3D bar chart of the given data.

        Args:
            data (np.array): The data to plot, which should be a 2D array representing the accumulated costs divided by
            the number of trials.
            title (str): Title of the plot.
            xlabel (str), ylabel (str), zlabel (str): Labels for the x, y, and z axes, respectively.
            :param title:
            :param data:
            :param zlabel:
            :param xlabel:
            :param ylabel:
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Create a grid of x, y coordinates covering the space of the data matrix
        xpos, ypos = np.meshgrid(range(data.shape[0]), range(data.shape[1]), indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = np.zeros_like(xpos)

        # The height of the bars should be the value from the data matrix
        dz = data.ravel()

        # The width and depth of the bars are set to 0.8 by default, but you can adjust these values as needed
        dx = dy = np.ones_like(dz) * 0.8

        # Creating the 3D bar chart
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True)

        # Setting labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)

        # Adding a color bar to indicate the scale of investment
        plt.colorbar(ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#5D8AA8'))

        plt.show()


if __name__ == "__main__":

    impact_simulator = ImpactBFS(P, A, B, M, tA, tB, tM, start, depth, trials)
    impact_simulator.run_simulation()
    impact_simulator.print_simulation_summary()
