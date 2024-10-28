import os
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
from utils import Utils
from input_data import A, B, M, tA, tB, tM, start, depth, trials
from input_data_api import input_data_instance
from simulation import Simulation
from plotter import Plotter

app = Flask(__name__)
CORS(app)

# Ensure the 'simulations' directory exists
if not os.path.exists('simulations'):
    os.makedirs('simulations')


@app.route('/run_simulation', methods=['POST'])
@cross_origin()
def run_simulation():
    data = request.json
    transition_matrix = data.get('transitionMatrix')
    best_cost_matrix = data.get('bestCostMatrix')
    worst_cost_matrix = data.get('worstCostMatrix')
    most_probable_cost_matrix = data.get('mostProbableCostMatrix')
    best_time_matrix = data.get('bestTimeMatrix')
    worst_time_matrix = data.get('worstTimeMatrix')
    most_probable_time_matrix = data.get('mostProbableTimeMatrix')
    print(transition_matrix)

    if not Utils.is_valid_matrix(transition_matrix):
        return jsonify({"error": "Invalid matrix format"}), 400

    # Update the matrices in input_data_instance
    input_data_instance.set_matrices(transition_matrix, best_cost_matrix, worst_cost_matrix, most_probable_cost_matrix,
                                     best_time_matrix, worst_time_matrix, most_probable_time_matrix)

    # Initialize the Simulation object with updated data
    simulation = Simulation(input_data_instance.P, input_data_instance.A, input_data_instance.B, input_data_instance.M,
                            input_data_instance.tA, input_data_instance.tB, input_data_instance.tM, start, depth, trials
                            )

    # Run the simulation
    simulation.run_simulation()

    # Generate timestamp-based folder name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_folder = os.path.join('simulations', timestamp)
    os.makedirs(output_folder, exist_ok=True)

    # Plotting and returning results
    plotter = Plotter(simulation)
    plotter.plot_histograms()
    plotter.plot_3d_bar_chart(simulation.cost_cum / trials)

    # Save plots to the output folder
    investment_histogram_path = os.path.join(output_folder, 'investment_histogram.png')
    time_histogram_path = os.path.join(output_folder, 'time_histogram.png')
    total_cost_histogram_path = os.path.join(output_folder, 'total_cost_histogram.png')
    bar_chart_path = os.path.join(output_folder, 'bar_chart.png')

    os.rename('static/investment_histogram.png', investment_histogram_path)
    os.rename('static/time_histogram.png', time_histogram_path)
    os.rename('static/total_cost_histogram.png', total_cost_histogram_path)
    os.rename('static/bar_chart.png', bar_chart_path)

    # Calculate KPIs and save to CSV
    mean_cost = np.sum(simulation.cost_cum) / simulation.trials
    mean_time = np.sum(simulation.time_cum) / simulation.trials
    mean_total_cost = np.sum(simulation.total_cum) / simulation.trials
    total_mean = simulation.all_sum / simulation.trials
    std_dev_total_cost = np.sqrt(simulation.all_sum_sqr / simulation.trials - (
            simulation.all_sum / simulation.trials) ** 2)

    kpis = {
        'Mean Cost': [mean_cost],
        'Mean Time': [mean_time],
        'Mean Total Cost': [mean_total_cost],
        'Total Mean': [total_mean],
        'Standard Deviation of Total Cost': [std_dev_total_cost]
    }
    kpis_df = pd.DataFrame(kpis)
    kpis_csv_path = os.path.join(output_folder, 'kpis.csv')
    kpis_df.to_csv(kpis_csv_path, index=False)

    # Return paths to the generated graphs
    return jsonify({"status": "Simulation completed", "graphs": [
        f"/simulations/{timestamp}/investment_histogram.png",
        f"/simulations/{timestamp}/time_histogram.png",
        f"/simulations/{timestamp}/total_cost_histogram.png",
        f"/simulations/{timestamp}/bar_chart.png"
    ], "kpis": f"/simulations/{timestamp}/kpis.csv"})


@app.route('/simulations/<path:filename>')
def serve_simulation_file(filename):
    directory = os.path.dirname(filename)
    filename = os.path.basename(filename)
    return send_from_directory(directory=os.path.join('simulations', directory), path=filename)


if __name__ == '__main__':
    app.run(debug=True)
