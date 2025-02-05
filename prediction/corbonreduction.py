import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

np.random.seed(42)

# Constants
emission_per_minute = 0.4  # kg CO2 emitted per minute of searching for parking

# Generate synthetic data
n_samples = 1000
search_times = np.random.randint(0, 30, size=n_samples)  # Search time in minutes (0 to 30 minutes)
emissions = search_times * emission_per_minute + np.random.normal(0, 0.02, size=n_samples)  # Carbon emissions

# Create a DataFrame
data = pd.DataFrame({
    'SearchTime': search_times,
    'CarbonEmissions': emissions
})

# Train-test split
X = data[['SearchTime']]
y = data['CarbonEmissions']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.4f} kg CO2")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(data['SearchTime'], data['CarbonEmissions'], alpha=0.5, label='Actual Data', color='blue')
plt.scatter(X_test, y_pred, color='red', label='Predicted Data', marker='x')
plt.title("Carbon Emissions Based on Search Time for Parking")
plt.xlabel("Search Time (minutes)")
plt.ylabel("Carbon Emissions (kg CO2)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
