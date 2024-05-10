import pandas as pd
from joblib import load

# Load the saved model and polynomial converter
load_model = load('G:\\Intern\\final_predict\\RandomForest_ModelHb.joblib')

# Function to predict concentration value given values from the 2nd to 4th columns
def predict_concentration(row_values):
    # Create a DataFrame with the user input
    user_input = pd.DataFrame({df.columns[1]: [row_values[0]], df.columns[2]: [row_values[1]], df.columns[3]: [row_values[2]]})
    
    # Predict concentration value using the loaded Random Forest model
    concentration_prediction = load_model.predict(user_input)
    
    return concentration_prediction[0]


# Load the dataset
df = pd.read_csv("G:\\Intern\\final_predict\\interpolatedHb.csv")  # Replace "your_csv_file.csv" with the path to your CSV file

# Get input from user for the input value
input_value = float(input("Enter the input value: "))  # Assuming the input value is a float

# Make prediction based on the input value
row_values = [input_value] * 3  # Assuming the input value should be used for all 3 columns
predicted_concentration = predict_concentration(row_values)

print("Predicted Concentration:", format(predicted_concentration,".2f"))