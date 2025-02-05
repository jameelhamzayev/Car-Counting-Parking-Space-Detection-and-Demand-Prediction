import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


np.random.seed(42)

hours_per_day = 24
days = 30
total_hours = hours_per_day * days


time_index = pd.date_range(start="2024-01-01", periods=total_hours, freq='H')


def generate_parking_demand(hour, day_of_week):
    base_demand = 50 + 10 * np.sin(hour * np.pi / 12) 
    weekend_increase = 20 if day_of_week >= 5 else 0  
    event_boost = 40 if np.random.random() > 0.95 else 0  
    return base_demand + weekend_increase + event_boost + np.random.normal(0, 5)

parking_data = []
for i in range(total_hours):
    day_of_week = time_index[i].weekday()
    hour_of_day = time_index[i].hour
    demand = generate_parking_demand(hour_of_day, day_of_week)
    parking_data.append([time_index[i], hour_of_day, day_of_week, demand])


parking_df = pd.DataFrame(parking_data, columns=["Timestamp", "Hour", "DayOfWeek", "ParkingDemand"])


parking_df['DemandClass'] = (parking_df['ParkingDemand'] > 70).astype(int)


X = parking_df[['Hour', 'DayOfWeek']] 
y = parking_df['DemandClass']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred) * 100  
print(f"Accuracy: {accuracy:.2f}%")


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.show()
