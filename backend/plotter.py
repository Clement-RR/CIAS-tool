import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Plotter:
    def __init__(self, simulation):
        self.simulation = simulation

    def plot_histograms(self):
        """
        Plots histograms for required investments, required time, and total cost based on simulation results.

        This method uses the Rice Rule to determine the bin size for histograms and plots separate
        histograms for accumulated costs, times, and total costs from the simulation.
        """
        binsize = int(2 * np.cbrt(self.simulation.trials))

        # Plot histogram of required investments
        plt.figure()
        plt.hist(self.simulation.cost_cum_list, bins=binsize, color='#C0504D')
        plt.title('Prediction of required investments (n = 20,000)')
        plt.xlabel('Monetary units')
        plt.ylabel('Frequency')
        plt.savefig('static/investment_histogram.png')
        plt.close()

        # Plot histogram of required time
        plt.figure()
        plt.hist(self.simulation.time_cum_list, bins=binsize, color='#6495ED')
        plt.title('Prediction of person-hours required for implementation (n = 20,000)')
        plt.xlabel('Person-hours')
        plt.ylabel('Frequency')
        plt.savefig('static/time_histogram.png')

        # Plot histogram of required total cost
        plt.figure()
        plt.hist(self.simulation.total_cum_list, bins=binsize, color='#9C9C9C')
        plt.title('Prediction of total cost (n = 20,000)')
        plt.xlabel('Monetary units')
        plt.ylabel('Frequency')
        plt.savefig('static/total_cost_histogram.png')
        plt.close()

        #plt.show()

        return ['static/investment_histogram.png', 'static/time_histogram.png', 'static/total_cost_histogram.png']

    def print_simulation_summary(self):
        """
        Prints a summary of the Monte Carlo simulation results, including
        mean values of costs, times, and total costs, and calculates the standard deviation
        for total costs.
        """
        # Calculating mean values
        mean_cost = np.sum(self.simulation.cost_cum) / self.simulation.trials
        mean_time = np.sum(self.simulation.time_cum) / self.simulation.trials
        mean_total_cost = np.sum(self.simulation.total_cum) / self.simulation.trials

        # Calculating total mean for all trials
        total_mean = self.simulation.all_sum / self.simulation.trials

        # Calculating standard deviation for total costs
        std_dev_total_cost = np.sqrt(self.simulation.all_sum_sqr / self.simulation.trials - (
                self.simulation.all_sum / self.simulation.trials) ** 2)

        # Displaying the results
        print("Monte Carlo Simulation Results:")
        print(f"Mean Cost: {mean_cost}")
        print(f"Mean Time: {mean_time}")
        print(f"Mean Total Cost: {mean_total_cost}")
        print(f"Total Mean: {total_mean}")
        print(f"Standard Deviation of Total Cost: {std_dev_total_cost}")

    def plot_3d_bar_chart(self, data, title='Heatmap of Expected Invest', xlabel='Affected', ylabel='Initiating',
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

        # The width and depth of the bars are set to 0.8 by default
        dx = dy = np.ones_like(dz) * 0.8

        # Mapping colors to the 'dz' values
        cmap = plt.cm.viridis  # You can choose any colormap you like
        colors = cmap(dz / np.max(dz))

        # Creating the 3D bar chart with color mapping
        bars = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors)

        # Setting labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)

        # Adding a color bar to indicate the scale of values
        plt.colorbar(bars, ax=ax, shrink=0.5, aspect=5)  # Adjust 'shrink' and 'aspect' as needed

        #plt.show()

        plt.savefig('static/bar_chart.png')
        return 'static/bar_chart.png'
