from flask import Flask, request, render_template, send_from_directory, url_for
import pandas as pd
import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt
import os
import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def calculate_SOM_size(data_size):
    five_sqrt_N_rule = 1+int(np.sqrt(5*np.sqrt(data_size)))
    return min(max(3, five_sqrt_N_rule),30)

def sanitize_column_names(df):
    df.columns = df.columns.str.replace('[^a-zA-Z0-9 ]', '', regex=True)
    df.columns = df.columns.str.replace(' ', '_')
    return df

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        csv_file = request.files['file']
        if not csv_file:
            return "No file"

        file_data = pd.read_csv(csv_file)
        sanitize_column_names(file_data)
        file_data = file_data.select_dtypes(include=[np.number]).dropna()

        # Train SOM
        som_size = calculate_SOM_size(file_data.shape[0])  # this defines the size of the SOM grid. It could be adjusted based on your specific data
        som = MiniSom(som_size, som_size, file_data.shape[1])
        som.train_random(file_data.values, 5000)  # you can adjust the number of iterations as needed

        # Plot SOM
        plt.figure(figsize=(10, 10))
        plt.pcolor(som.distance_map().T, cmap='terrain')  # plotting the distance map as background
        plt.colorbar()

        # Save plot as png
        timestamp = time.strftime('%Y%m%d%H%M%S')
        filename = f'static/plot_{timestamp}.png'
        plt.savefig(filename)
        plt.close()

        # Create DataFrame of the SOM weights
        weights = som.get_weights().reshape(-1, file_data.shape[1])
        weights_df = pd.DataFrame(weights, columns=file_data.columns)
        weights_df.to_csv('static/weights.csv', index=False)
        # Generate a plot for each variable
        variable_plots = []
        for column, i in zip(file_data.columns, range(len(file_data.columns))):
            plt.figure(figsize=(10, 10))
            feature_weights = som.get_weights()[:, :, i]
            plt.imshow(feature_weights, cmap='coolwarm')
            plt.title(column)
            #feature_weights = som.get_weights()[:, :, file_data.columns.get_loc(column)]
            #feature_activations = np.tensordot(feature_data, feature_weights, axes=([1], [2]))
            # Plot
            #plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
            #plt.colorbar()
            #plt.pcolor(feature_activations, cmap='Reds')  # plotting feature activations
            #plt.hist(feature_data)
            #plt.hist(feature_som, alpha=0.5)
            filename = f'static/plot_{column}_{timestamp}.png'
            plt.savefig(filename)
            plt.close()
            variable_plots.append(url_for('static', filename=f'plot_{column}_{timestamp}.png'))

        return render_template('view.html', graph_file=url_for('static', filename=f'plot_{timestamp}.png'), 
                             variable_plots=variable_plots)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    app.run(debug=True)
