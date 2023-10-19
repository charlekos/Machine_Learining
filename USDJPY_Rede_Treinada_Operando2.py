import MetaTrader5 as mt5
import pandas as pd
import pickle
import numpy as np
import time
from termcolor import colored
from tabulate import tabulate
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
import os

# Function to clear the terminal screen
def clear_screen():
    if os.name == 'posix':
        os.system('clear')  # For Linux/Unix
    elif os.name == 'nt':
        os.system('cls')  # For Windows

# Specify the scaling factor to convert predictions into points (customize this value)
points_per_unit = 0.01  # Adjust this value according to your instrument

# Initialize an empty list to store data
data = []

# Initialize MetaTrader 5
mt5.initialize()

# Specify the trading symbol and timeframe
symbol = "USDJPY"
timeframe = mt5.TIMEFRAME_M1

# Load the pre-trained neural network model
model_filename = f'{symbol}_modelo_rede_neural.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Define the model architecture to match the input data shape
model = keras.Sequential([
    keras.layers.LSTM(units=64, input_shape=(3, 1)),
    keras.layers.Dense(units=1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

while True:
    clear_screen()

    # Get the current data from MetaTrader 5
    current_data = mt5.copy_rates_from_pos(symbol, timeframe, 0, 2)  # Fetch 2 bars to calculate the prediction
    current_open = current_data[0]['open']
    current_close = current_data[0]['close']
    current_tick_volume = current_data[0]['tick_volume']

    # Calculate the time for the prediction
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    prediction_time = (datetime.now() + timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M:%S')

    # Prepare data for prediction with the correct shape
    input_data = np.array([[[current_open], [current_close], [current_tick_volume]]])

    # Make a prediction for the next minute
    prediction = model.predict(input_data)[0][0]

    # Convert the prediction to points
    prediction_in_points = prediction * points_per_unit * 400

    # Fetch the close price one minute into the future
    future_data = mt5.copy_rates_from_pos(symbol, timeframe, 1, 2)
    future_close = future_data[0]['close']

    # Determine if the prediction was correct
    if((prediction_in_points <= 0 and current_close <= future_close ) or (prediction_in_points >= 0 and current_close >= future_close )):
        
        prediction_correct = True
    else:
        prediction_correct = False

    # Determine the color based on the prediction result
    if prediction_correct:
        color = 'green'
    else:
        color = 'red'

    # Create a list with colored values for the entire row, including the predicted value
    colored_row = [
        colored(current_open, color),
        colored(current_close, color),
        colored(future_close, color),
        colored(current_tick_volume, color),
        colored(prediction_in_points, color),
        colored(current_time, color),
        colored(prediction_time, color),
        colored("Correct" if prediction_correct else "Incorrect", color)
    ]

    # Append the colored row to the data
    data.append(colored_row)

    # Keep only the last ten rows of data
    data = data[-10:]

    # Create a table from the DataFrame and print it with colored rows
    headers = ["Current Open", "Current Close", "Future Close", "Tick Volume", "Next Minute Prediction (Points)", "Local Time", "Prediction Time", "Prediction Result"]
    table = pd.DataFrame(data, columns=headers)

    # Print the table with colored rows
    print(tabulate(table, headers=headers, tablefmt='grid'))

    # Print the prediction value in points
    print("Previsao:")
    print(prediction_in_points)
    print("Pontuacao:")
    print(future_close)

    # Wait for a specific interval (1 minute) before the next update
    time.sleep(60)

# Deinitialize MetaTrader 5 before restarting the loop
mt5.shutdown()
