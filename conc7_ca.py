from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

app = Flask(__name__)

# Read the CSV file into a DataFrame (Update file path)
df = pd.read_csv("G:\\Intern\\xlsx_files\\CA_Model01_(1)_sample.csv")

# Calculate the average and standard deviation of 'aval' and 'bval'
df['average'] = (df['min_values'] + df['max_values']) / 2
df['std_dev'] = df[['min_values', 'max_values']].std(axis=1)

# Initialize the decision tree regressor as the base estimator for AdaBoost
base_estimator = DecisionTreeRegressor(max_depth=5)

# Initialize and fit the AdaBoost regression model with a fixed random seed
ada_model = AdaBoostRegressor(base_estimator=base_estimator, n_estimators=50, learning_rate=0.0001, random_state=42)
ada_model.fit(df[['average']], df['concentration'])

# Define the pipeline with preprocessing and Lasso regression
lasso_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', Lasso(alpha=0.1))
])

# Define a wider range of alpha values for Lasso regression
alphas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10]

# Define the parameter grid for GridSearchCV
param_grid = {'lasso__alpha': alphas}

# Reduce the number of folds for cross-validation to 5
grid_search = GridSearchCV(lasso_pipeline, param_grid, cv=5)

# Fit the grid search to the data
grid_search.fit(df[['average', 'std_dev']], df['concentration'])

# Get the best Lasso model with the optimal alpha value
best_lasso_model = grid_search.best_estimator_


def predict_conc(input_value):
    # Check if input_value is within the range of 'aval' and 'bval'
    for i, row in df.iterrows():
        if row['min_values'] <= input_value <= row['max_values']:
            return ada_model.predict([[input_value]])
    # If input_value is out of range, use Lasso regression
    return best_lasso_model.predict([[input_value, 0]])  # Assuming a default std_dev of 0 for out-of-range values

@app.route("/")
def root():
    with open("G:\\Intern\\design.html") as file:
        return file.read()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_value = float(request.form["input_value"])
        predicted_conc = predict_conc(input_value)
        return jsonify({"prediction": predicted_conc[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
