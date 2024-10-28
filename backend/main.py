"""
This module runs a simulation to analyze the impact of changes within a system, plots the results,
and prints a summary. It uses predefined input data and the Simulation and Plotter classes.
"""

from simulation import Simulation
from plotter import Plotter
from input_data import P, A, B, M, tA, tB, tM, start, depth, trials


def main():
    """
    Initializes and runs the simulation, then plots and prints the results.
    """
    # Initialize the Simulation object
    simulation = Simulation(P, A, B, M, tA, tB, tM, start, depth, trials)

    # Run the simulation
    simulation.run_simulation()

    # Plotting and printing results
    plotter = Plotter(simulation)
    plotter.plot_histograms()
    plotter.plot_3d_bar_chart(simulation.cost_cum / trials)
    plotter.print_simulation_summary()


if __name__ == "__main__":
    main()
