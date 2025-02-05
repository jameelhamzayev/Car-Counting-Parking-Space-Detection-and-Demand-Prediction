import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

initial_cars_2024 = 1738940
annual_growth_rate = 0.056

years = np.arange(2024, 2030)
num_years = len(years)
car_numbers = []

for year in years:
    cars_this_year = initial_cars_2024 * (1 + annual_growth_rate) ** (year - 2024)
    car_numbers.append(cars_this_year)

parking_demand = np.array(car_numbers)

df = pd.DataFrame({
    'Year': years,
    'Number of Cars': car_numbers,
    'Parking Demand': parking_demand
})

plt.figure(figsize=(10, 6))
plt.plot(df['Year'], df['Number of Cars'], label='Predicted Number of Cars', marker='o', color='b')
plt.plot(df['Year'], df['Parking Demand'], label='Predicted Parking Demand', linestyle='--', color='r')
plt.title("Predicted Growth in Number of Cars and Parking Demand (2024-2029)")
plt.xlabel("Year")
plt.ylabel("Number of Cars / Parking Demand (million)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

print(df)
