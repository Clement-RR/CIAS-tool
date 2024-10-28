"""
This module performs data analysis and visualization based on beta distribution,
including statistical analysis and 3D plotting of impacts.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm
from scipy.sparse import random as sprandom
# The import of Axes3D is not directly used but is necessary for 3D plotting.
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import


class DataAnalyzer:
    """
    A class to analyze data based on beta distribution.

    Attributes:
        trials (int): Number of trials for the data generation.
        data (ndarray): Data generated from the beta distribution.
    """

    def __init__(self, trials=20000):
        """
        Initializes the DataAnalyzer with a specified number of trials.

        Args:
            trials (int): Number of trials to generate data. Defaults to 20000.
        """
        self.trials = trials
        self.data = 100000 * beta.rvs(2, 5, size=trials)

    def calculate_statistics(self):
        """
        Calculates and returns statistical measures of the data.

        Returns:
            tuple: Mean, standard deviation, and 90th percentile of the data.
        """
        mean_value = np.mean(self.data)
        std_dev = np.sqrt(np.var(self.data))
        percentile_90 = np.percentile(self.data, 90)
        return mean_value, std_dev, percentile_90


class Plotter:
    """
    A class for plotting data analysis results.
    """

    @staticmethod
    def plot_histogram_and_cdf(data, trials):
        """
        Plots the histogram and cumulative density function (CDF) of the given data.

        Args:
            data (ndarray): The data to plot.
            trials (int): Number of trials, used to calculate binsize.
        """
        binsize = int(2 * trials ** (1/3))
        fig, ax1 = plt.subplots()

        # Histogram
        y1, x1, _ = ax1.hist(data, bins=binsize, alpha=0.75)
        ax1.set_xlabel('EUR')
        ax1.set_ylabel('Frequency', color='tab:blue')

        # CDF
        ax2 = ax1.twinx()
        y2, x2 = np.histogram(data, bins=binsize, density=True)
        cdf = np.cumsum(y2) * np.diff(x2)
        ax2.plot(x2[:-1], cdf, color='red', linestyle='-', linewidth=1.5)
        ax2.set_ylabel('CDF', color='tab:red')

        plt.title('Simulation of total cost')
        plt.show()

    @staticmethod
    def plot_3d_bar_chart(matrix):
        """
        Plots a 3D bar chart of the given matrix.

        Args:
            matrix (ndarray): The matrix to plot in a 3D bar chart.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xpos, ypos = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]), indexing="ij")
        xpos, ypos = xpos.ravel(), ypos.ravel()
        zpos = np.zeros_like(xpos)
        dx = dy = np.ones_like(zpos) * 0.8
        dz = matrix.ravel()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True)
        ax.set_xlabel('Affected')
        ax.set_ylabel('Initiating')
        ax.set_zlabel('Impact (EUR)')
        plt.title('Change impact heatmap')
        plt.show()


def main():
    """
    Main function to execute data analysis and plotting routines.
    """
    analyzer = DataAnalyzer()
    mean_value, std_dev, percentile_90 = analyzer.calculate_statistics()
    print(f"Mean: {mean_value:.2f}, Std Dev: {std_dev:.2f}, 90th Percentile: {percentile_90:.2f}")

    Plotter.plot_histogram_and_cdf(analyzer.data, analyzer.trials)

    # Generating and plotting edge impact
    edge_impact = 8000 * sprandom(10, 10, density=0.5, data_rvs=norm().rvs).A
    edge_impact[edge_impact < 0] = 0
    Plotter.plot_3d_bar_chart(edge_impact)


if __name__ == "__main__":
    main()
