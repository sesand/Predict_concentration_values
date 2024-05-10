import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from joblib import dump, load

# Load the dataset
df = pd.read_csv("C:\\aathi obusre\\interpolatedca.csv")

# Extract only the columns needed for training
X_cols = ['Ca SET1', 'Ca SET2', 'Ca SET3']
X = df[X_cols]

# Handle missing values if any
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split the data into features (X) and target variable (y)
y = df['concentration']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Polynomial conversion and training for Polynomial Regression
polynomial_converter = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = polynomial_converter.fit_transform(X_train)
X_test_poly = polynomial_converter.transform(X_test)

polymodel = LinearRegression()
polymodel.fit(X_train_poly, y_train)

# Predictions and Evaluation for Polynomial Regression
y_pred_poly = polymodel.predict(X_test_poly)
poly_mae = mean_absolute_error(y_test, y_pred_poly)
poly_mse = mean_squared_error(y_test, y_pred_poly)
poly_rmse = np.sqrt(poly_mse)

print("Polynomial Regression Evaluation:")
print("Mean Absolute Error (MAE):", poly_mae)
print("Mean Squared Error (MSE):", poly_mse)
print("Root Mean Squared Error (RMSE):", poly_rmse)

# Saving the Polynomial Regression model
dump(polymodel, 'Hb_Model5.joblib')
dump(polynomial_converter, 'Hb_Converter5.joblib')

# Model Selection and Hyperparameter Tuning
# Example: Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Best model
best_rf_model = grid_search.best_estimator_

# Cross-validation
cv_scores = cross_val_score(best_rf_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse_scores = np.sqrt(-cv_scores)
print("Cross-Validation RMSE Scores:", cv_rmse_scores)
print("Mean CV RMSE:", cv_rmse_scores.mean())

# Final evaluation on test set
y_pred_rf = best_rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(rf_mse)

print("Random Forest Regression Evaluation:")
print("Mean Absolute Error (MAE):", rf_mae)
print("Mean Squared Error (MSE):", rf_mse)
print("Root Mean Squared Error (RMSE):", rf_rmse)

# Saving the Random Forest Regression model
dump(best_rf_model, 'RandomForest_ModelCa.joblib')
