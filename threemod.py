import pandas as pd
from joblib import load

# Load the saved models
ca_model = load('G:\\Intern\\final_predict\\RandomForest_ModelCa.joblib')
hb_model = load('G:\\Intern\\final_predict\\RandomForest_ModelHb.joblib')
gl_model = load('G:\\Intern\\final_predict\\RandomForest_ModelGl.joblib')  # Assuming the same model for both hb and gl

# Load the dataset
df_ca = pd.read_csv("G:\\Intern\\final_predict\\interpolatedca.csv")
df_hb = pd.read_csv("G:\\Intern\\final_predict\\interpolatedHb.csv")
df_gl = pd.read_csv("G:\\Intern\\final_predict\\interpolatedgl.csv")  # Assuming the same dataset for hb and gl

# Function to predict concentration value given values from the 2nd to 4th columns
def predict_concentration(model, df, row_values):
    # Create a DataFrame with the user input
    user_input = pd.DataFrame({df.columns[1]: [row_values[0]], df.columns[2]: [row_values[1]], df.columns[3]: [row_values[2]]})
    
    # Predict concentration value using the loaded Random Forest model
    concentration_prediction = model.predict(user_input)
    
    return concentration_prediction[0]

    # Create a DataFrame with the user input
# Ask the user which model to predict
model_choice = input("Enter the model to predict (ca/hb/gl): ")

# Get input from user for the input value
input_value = float(input("Enter the input value: "))  # Assuming the input value is a float

# Make prediction based on the input value and chosen model
if model_choice == "ca":
    predicted_concentration = predict_concentration(ca_model, df_ca, [input_value] * 3)
elif model_choice == "hb":
    predicted_concentration = predict_concentration(hb_model, df_hb, [input_value] * 3)
elif model_choice == "gl":
    predicted_concentration = predict_concentration(gl_model, df_gl, [input_value] * 3)
else:
    print("Invalid model choice. Please choose 'ca', 'hb', or 'gl'.")

print("Predicted Concentration:", format(predicted_concentration, ".2f"))
