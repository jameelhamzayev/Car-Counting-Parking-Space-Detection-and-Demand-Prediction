import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

np.random.seed(42)


hours_per_day = 24
days = 30
total_hours = hours_per_day * days


time_index = pd.date_range(start="2024-01-01", periods=total_hours, freq='H')


def generate_parking_demand(hour, day_of_week):
    base_demand = 600 + 10 * np.sin(hour * np.pi / 12)  
    weekend_increase = 20 if day_of_week >= 5 else 0  
    event_boost = 40 if np.random.random() > 0.95 else 0  
    return base_demand + weekend_increase + event_boost + np.random.normal(0, 5)

parking_data = []
for i in range(total_hours):
    day_of_week = time_index[i].weekday()
    hour_of_day = time_index[i].hour
    demand = generate_parking_demand(hour_of_day, day_of_week)
    parking_data.append([time_index[i], hour_of_day, day_of_week, demand])

# Create DataFrame
parking_df = pd.DataFrame(parking_data, columns=["Timestamp", "Hour", "DayOfWeek", "ParkingDemand"])

# Feature preprocessing
X = parking_df[['Hour', 'DayOfWeek']]  
y = parking_df['ParkingDemand']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")


plt.figure(figsize=(10, 6))
plt.plot(y_test.values[:100], label="Actual", marker='o')
plt.plot(y_pred[:100], label="Predicted", marker='x')
plt.title("Actual vs Predicted Parking Demand in Heydar Aliyev International Airport Terminal 1 ")
plt.xlabel("Time (in 100 days)(Start time 01.01.2024)")
plt.ylabel("Parking Demand")
plt.legend()
plt.show()
