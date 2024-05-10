"""import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
from matplotlib.ticker import ScalarFormatter
def plot_data(csv_file, subplot_title, ax):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Calculate the average of aval and bval
    df['average'] = (df['aval'] + df['bval']) / 2

    # Scatter plot
    scatter = ax.scatter(df['con'], df['average'], label=subplot_title)

    # Calculate y-axis limits
    y_min = df['average'].min()
    y_max = df['average'].max()

    # Set y-axis limits with some buffer
    ax.set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    # Set y-axis tick formatter to display full values
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

    # Add mplcursors to display data values on hover
    mplcursors.cursor(scatter)

    # Add subplot title
    ax.set_title(subplot_title)

# Create a figure and subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Plot data from each CSV file
plot_data('C:\\aathi obusre\\Model_GI.csv', 'G1 Data', ax1)
plot_data('C:\\aathi obusre\\Model_CA.csv', 'CA Data', ax2)
plot_data('C:\\aathi obusre\\Model_HB.csv', 'HB Data', ax3)

# Add labels and title
fig.text(0.5, 0.04, 'con', ha='center')
fig.text(0.04, 0.5, 'Average', va='center', rotation='vertical')
fig.suptitle('Scatter Plots')

# Show the plot
plt.grid(True)
plt.legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust subplot spacing
plt.show()"""

import pandas as pd
import numpy as np

# Given values
min_value = 12923	
max_value = 13370
# Generate random sets of three values within the range
sets_of_values = []
for _ in range(75):
    set_values = np.random.randint(min_value, max_value + 1, size=3)
    sets_of_values.append(set_values)

# Create a DataFrame from the sets of values
df = pd.DataFrame(sets_of_values, columns=['Value 1', 'Value 2', 'Value 3'])

# Save the DataFrame to an Excel file
df.to_excel('sets_of_values.xlsx', index=False)

"""
import pandas as pd
from joblib import load

# Load the saved polynomial regression model
load_model = load('C:\\aathi obusre\\Polynomial_Regression_Model2.joblib')

# Function to predict concentration value using the polynomial regression model
def predict_concentration(input_value):
    # Create a DataFrame with the user input
    user_input = pd.DataFrame({'Ca SET 1': [input_value], 'Ca SET 2': [input_value], 'Ca SET 3': [input_value]})
    
    # Predict concentration value using the loaded model
    concentration_prediction = load_model.predict(user_input)
    
    return concentration_prediction[0]

# Load the modified dataset from a CSV file
modified_dataset = pd.read_csv('C:\\aathi obusre\\modified_dataset.csv')  # Replace 'modified_dataset.csv' with the path to your modified dataset CSV file

# Function to check if the input value lies within any range and return the concentration value if it does
def check_and_predict(input_value, modified_dataset):
    for index, row in modified_dataset.iterrows():
        if row['min_value'] <= input_value <= row['max_value']:
            return row['concentration']
    return None

# Get input from the user for the input value
input_value = float(input("Enter the input value: "))  # Assuming the input value is a float

# Check if the input value lies within any range in the dataset
concentration = check_and_predict(input_value, modified_dataset)

if concentration is not None:
    print("Predicted concentration value based on range:", concentration)
else:
    # Predict concentration value using the polynomial regression model
    predicted_concentration = predict_concentration(input_value)
    print("Predicted concentration value using polynomial regression model:", predicted_concentration)
"""
"""
import pandas as pd

# Load the original dataset from the CSV file
original_dataset = pd.read_csv('C:\\aathi obusre\\interploatedchanges.csv')  # Replace 'original_dataset.csv' with the path to your original dataset CSV file

# Function to modify the dataset to include minimum and maximum values
def modify_dataset(original_dataset):
    modified_data = []
    for index, row in original_dataset.iterrows():
        min_value = min(row.iloc[1], row.iloc[3])  # Calculate the minimum value between the 2nd and 3rd columns
        max_value = max(row.iloc[1], row.iloc[3])  # Calculate the maximum value between the 2nd and 3rd columns
        concentration = row['concentration']
        modified_data.append({'min_value': min_value, 'max_value': max_value, 'concentration': concentration})
    return pd.DataFrame(modified_data)

# Modify the dataset to include minimum and maximum values
modified_dataset = modify_dataset(original_dataset)

# Save the modified dataset to a new CSV file
modified_dataset.to_csv('modified_dataset.csv', index=False)

"""


