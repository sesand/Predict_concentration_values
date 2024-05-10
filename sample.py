from flask import Flask, render_template, request
import pandas as pd
from joblib import load

app = Flask(__name__)

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

@app.route("/")
def root():
    with open("G:\\Intern\\final_predict\\threemodel.html") as file:
        return file.read()

@app.route('/predict', methods=['POST'])
def predict():
    print("Received request...")
    model_choice = request.form['model']
    input_values = [float(val.strip()) for val in request.form['input_value'].split(',')]
    print("Model Choice:", model_choice)
    print("Input Values:", input_values)
    
    # Make prediction based on the input value and chosen model
    if model_choice == "ca":
        predicted_concentrations = [predict_concentration(ca_model, df_ca, [val] * 3) for val in input_values]
    elif model_choice == "hb":
        predicted_concentrations = [predict_concentration(hb_model, df_hb, [val] * 3) for val in input_values]
    elif model_choice == "gl":
        predicted_concentrations = [predict_concentration(gl_model, df_gl, [val] * 3) for val in input_values]
    else:
        return "Invalid model choice. Please choose 'ca', 'hb', or 'gl'."
    
    print("Predicted Concentrations:", predicted_concentrations)
    
    # Create a string to display results
    result_string = "<h3>Predicted Concentrations:</h3><ul>"
    for input_val, pred_conc in zip(input_values, predicted_concentrations):
        result_string += f"<li>Input Value: {input_val}, Predicted Concentration: {pred_conc:.2f}</li>"
    result_string += "</ul>"
    
    # Render results on the same page using JavaScript
    return f"""
    <script>
        document.getElementById("prediction_results").innerHTML = `{result_string}`;
    </script>
    """

if __name__ == '__main__':
    app.run(debug=True)
