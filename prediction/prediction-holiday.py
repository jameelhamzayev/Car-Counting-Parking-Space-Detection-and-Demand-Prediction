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

def generate_parking_demand(hour, day_of_week, is_special_day):
    base_demand = 100 + 10 * np.sin(hour * np.pi / 12)
    weekend_increase = 20 if day_of_week >= 5 else 0
    event_boost = 40 if is_special_day else 0
    return base_demand + weekend_increase + event_boost + np.random.normal(0, 5)

special_days = {
    '2024-01-01': 'New Year',
    '2024-02-14': '14 February',
    '2024-03-21': 'Novruz Holiday',
    '2024-11-11': 'COP29',
    '2024-12-25': 'Christmas',
}

parking_data = []
for i in range(total_hours):
    day = time_index[i].date()
    day_of_week = time_index[i].weekday()
    hour_of_day = time_index[i].hour
    is_special_day = 1 if str(day) in special_days else 0
    demand = generate_parking_demand(hour_of_day, day_of_week, is_special_day)
    parking_data.append([time_index[i], hour_of_day, demand, is_special_day])

parking_df = pd.DataFrame(parking_data, columns=["Timestamp", "Hour", "ParkingDemand", "IsSpecialDay"])

average_parking_by_hour = parking_df.groupby(["Hour", "IsSpecialDay"])["ParkingDemand"].mean().reset_index()

X = average_parking_by_hour[['Hour', 'IsSpecialDay']]
y = average_parking_by_hour['ParkingDemand']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")

tolerance = 10
correct_predictions = np.sum((y_pred >= (y_test - tolerance)) & (y_pred <= (y_test + tolerance)))
accuracy = (correct_predictions / len(y_test)) * 100
print(f"Accuracy: {accuracy:.2f}%")

plt.figure(figsize=(10, 6))
predicted_demand = model.predict(np.array([[hour, 0] for hour in range(24)]))
predicted_special_demand = model.predict(np.array([[hour, 1] for hour in range(24)]))
plt.plot(np.arange(24), predicted_demand, label="Predicted (Regular Days)", marker='x', linestyle='-', color='r')
plt.plot(np.arange(24), predicted_special_demand, label="Predicted (Special Days)", marker='o', linestyle='--', color='g')
plt.scatter(np.arange(24), average_parking_by_hour[average_parking_by_hour['IsSpecialDay'] == 0]['ParkingDemand'],
            label="Actual (Regular Days)", color='b', marker='o')
plt.scatter(np.arange(24), average_parking_by_hour[average_parking_by_hour['IsSpecialDay'] == 1]['ParkingDemand'],
            label="Actual (Special Days)", color='orange', marker='x')
plt.xticks(np.arange(24), [f'{i}:00' for i in range(24)], rotation=45)
plt.title("Actual vs Predicted Parking Demand by Hour of Day (H.A Airport Terminal 1)")
plt.xlabel("Hour of the Day")
plt.ylabel("Average Parking Demand")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
