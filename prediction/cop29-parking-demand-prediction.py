import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)

hours_per_day = 24

def generate_future_parking_demand(hour, is_special_day):
    base_demand = 270 + 10 * np.sin(hour * np.pi / 12)
    event_boost = 100 if is_special_day else 0
    return base_demand + event_boost + np.random.normal(0, 5)

event_date = '2024-11-15'
event_day = pd.to_datetime(event_date).date()
special_days = {event_date: 'COP29'}

future_parking_data = []
for hour in range(hours_per_day):
    is_special_day = 1 if event_date in special_days else 0
    demand = generate_future_parking_demand(hour, is_special_day)
    future_parking_data.append([event_day, hour, demand, is_special_day])

future_parking_df = pd.DataFrame(future_parking_data, columns=["Date", "Hour", "ParkingDemand", "IsSpecialDay"])

X_future = future_parking_df[['Hour', 'IsSpecialDay']]
model = RandomForestRegressor(n_estimators=100, random_state=42)

previous_parking_data = pd.DataFrame(np.array([[h, 0] for h in range(24)]), columns=['Hour', 'IsSpecialDay'])
previous_parking_data['ParkingDemand'] = [generate_future_parking_demand(h, 0) for h in range(24)]
model.fit(previous_parking_data[['Hour', 'IsSpecialDay']], previous_parking_data['ParkingDemand'])

future_predictions = model.predict(X_future)

plt.figure(figsize=(10, 6))
plt.plot(range(hours_per_day), future_predictions, label="Predicted Parking Demand for COP29", marker='o', linestyle='-', color='g')
plt.xticks(range(hours_per_day), [f'{i}:00' for i in range(hours_per_day)], rotation=45)
plt.title("Predicted Parking Demand for COP29")
plt.xlabel("Hour of the Day")
plt.ylabel("Predicted Parking Demand")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

for hour, demand in enumerate(future_predictions):
    print(f"Hour {hour}: Predicted Parking Demand = {demand:.2f}")
