import numpy as np
import pandas as pd

# Read the CSV file
data = pd.read_csv("C:\\aathi obusre\\ca.csv")

# Assuming your CSV file has columns named "concentration", "Ca SET 1", "Ca SET 2", and "Ca SET 3"
concentration = data["concentration"].values
ca_set1 = data["Ca SET 1"].values
ca_set2 = data["Ca SET 2"].values
ca_set3 = data["Ca SET 3"].values

# Interpolation
num_interpolations = 25 # Number of interpolated samples between each pair of existing data points

interpolated_concentration = []
interpolated_ca_set1 = []
interpolated_ca_set2 = []
interpolated_ca_set3 = []

for i in range(len(concentration) - 1):
    # Generate interpolated values
    conc_interp = np.linspace(concentration[i], concentration[i + 1], num_interpolations + 2)[1:-1]
    ca_set1_interp = np.linspace(ca_set1[i], ca_set1[i + 1], num_interpolations + 2)[1:-1]
    ca_set2_interp = np.linspace(ca_set2[i], ca_set2[i + 1], num_interpolations + 2)[1:-1]
    ca_set3_interp = np.linspace(ca_set3[i], ca_set3[i + 1], num_interpolations + 2)[1:-1]

    # Append interpolated values
    interpolated_concentration.extend(conc_interp)
    interpolated_ca_set1.extend(ca_set1_interp)
    interpolated_ca_set2.extend(ca_set2_interp)
    interpolated_ca_set3.extend(ca_set3_interp)

# Create a new DataFrame for the interpolated data
interpolated_data = pd.DataFrame({
    "concentration": interpolated_concentration,
    "Ca SET1": interpolated_ca_set1,
    "Ca SET2": interpolated_ca_set2,
    "Ca SET3": interpolated_ca_set3
})

# Save the interpolated data to a new CSV file
interpolated_data.to_csv("interpolatedca.csv", index=False)