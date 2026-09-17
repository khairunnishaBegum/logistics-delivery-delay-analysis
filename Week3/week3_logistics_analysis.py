import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

# Create simulated logistics data
n = 150

df = pd.DataFrame({
    "Order_ID": range(1001, 1001 + n),
    "Distance_km": np.round(np.random.uniform(1, 25, n), 2),
    "Traffic": np.random.choice(
        ["Low", "Medium", "High", "Jam"], n
    ),
    "Weather": np.random.choice(
        ["Clear", "Rain", "Storm"], n
    )
})

# Time added because of traffic
traffic_time = {
    "Low": 5,
    "Medium": 10,
    "High": 18,
    "Jam": 25
}

# Time added because of weather
weather_time = {
    "Clear": 0,
    "Rain": 7,
    "Storm": 12
}

# Calculate delivery time
df["Delivery_Time_min"] = (
    15
    + df["Distance_km"] * 2
    + df["Traffic"].map(traffic_time)
    + df["Weather"].map(weather_time)
    + np.random.randint(-5, 6, n)
)

df["Delivery_Time_min"] = (
    df["Delivery_Time_min"]
    .round()
    .astype(int)
)

# Calculate delivery cost
df["Delivery_Cost"] = (
    30 + df["Distance_km"] * 4
).round(2)

# Generate customer ratings
df["Customer_Rating"] = np.round(
    np.random.uniform(3.0, 5.0, n), 1
)

# Mark deliveries as delayed or on time
df["Delivery_Status"] = np.where(
    df["Delivery_Time_min"] > 70,
    "Delayed",
    "On Time"
)

# Display basic information
print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nStatistical Summary:")
print(df.describe())

# Central tendency
print("\nAverage Values:")
print("Average Distance:",
      round(df["Distance_km"].mean(), 2), "km")

print("Average Delivery Time:",
      round(df["Delivery_Time_min"].mean(), 2), "minutes")

print("Average Delivery Cost:",
      round(df["Delivery_Cost"].mean(), 2))

print("Average Customer Rating:",
      round(df["Customer_Rating"].mean(), 2))

# Traffic analysis
traffic_avg = df.groupby(
    "Traffic"
)["Delivery_Time_min"].mean().sort_values()

print("\nAverage Delivery Time by Traffic:")
print(traffic_avg)

# Weather analysis
weather_avg = df.groupby(
    "Weather"
)["Delivery_Time_min"].mean().sort_values()

print("\nAverage Delivery Time by Weather:")
print(weather_avg)

# Delivery status
status_count = df["Delivery_Status"].value_counts()

print("\nDelivery Status:")
print(status_count)

delayed_percentage = (
    (df["Delivery_Status"] == "Delayed").sum()
    / len(df)
) * 100

print("\nDelayed Delivery Percentage:",
      round(delayed_percentage, 2), "%")

# Correlation analysis
corr = df[
    [
        "Distance_km",
        "Delivery_Time_min",
        "Delivery_Cost",
        "Customer_Rating"
    ]
].corr()

print("\nCorrelation:")
print(corr)

# Distance range trend analysis
df["Distance_Range"] = pd.cut(
    df["Distance_km"],
    bins=[0, 5, 10, 15, 20, 25],
    labels=[
        "1-5 km",
        "6-10 km",
        "11-15 km",
        "16-20 km",
        "21-25 km"
    ]
)

trend = df.groupby(
    "Distance_Range",
    observed=False
)["Delivery_Time_min"].mean()

print("\nAverage Delivery Time by Distance Range:")
print(trend)

# Traffic and weather bottleneck analysis
bottleneck = df.groupby(
    ["Traffic", "Weather"]
)["Delivery_Time_min"].mean().sort_values(
    ascending=False
)

print("\nTraffic and Weather Analysis:")
print(bottleneck)

# Average cost by traffic
cost_traffic = df.groupby(
    "Traffic"
)["Delivery_Cost"].mean().sort_values(
    ascending=False
)

print("\nAverage Delivery Cost by Traffic:")
print(cost_traffic)

# ---------------- VISUALIZATIONS ----------------

# 1. Delivery time distribution
plt.figure(figsize=(8, 5))
plt.hist(
    df["Delivery_Time_min"],
    bins=10
)
plt.xlabel("Delivery Time (minutes)")
plt.ylabel("Number of Orders")
plt.title("Distribution of Delivery Time")
plt.show()

# 2. Distance distribution
plt.figure(figsize=(8, 5))
plt.hist(
    df["Distance_km"],
    bins=10
)
plt.xlabel("Distance (km)")
plt.ylabel("Number of Orders")
plt.title("Distribution of Delivery Distance")
plt.show()

# 3. Average delivery time by traffic
plt.figure(figsize=(8, 5))
traffic_avg.plot(kind="bar")
plt.xlabel("Traffic Level")
plt.ylabel("Average Delivery Time (minutes)")
plt.title("Average Delivery Time by Traffic")
plt.xticks(rotation=0)
plt.show()

# 4. Average delivery time by weather
plt.figure(figsize=(8, 5))
weather_avg.plot(kind="bar")
plt.xlabel("Weather")
plt.ylabel("Average Delivery Time (minutes)")
plt.title("Average Delivery Time by Weather")
plt.xticks(rotation=0)
plt.show()

# 5. Distance vs delivery time
plt.figure(figsize=(8, 5))
plt.scatter(
    df["Distance_km"],
    df["Delivery_Time_min"]
)
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (minutes)")
plt.title("Distance vs Delivery Time")
plt.show()

# 6. Distance vs delivery cost
plt.figure(figsize=(8, 5))
plt.scatter(
    df["Distance_km"],
    df["Delivery_Cost"]
)
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Cost")
plt.title("Distance vs Delivery Cost")
plt.show()

# 7. Delivery status
plt.figure(figsize=(7, 5))
status_count.plot(kind="bar")
plt.xlabel("Delivery Status")
plt.ylabel("Number of Orders")
plt.title("On-Time vs Delayed Deliveries")
plt.xticks(rotation=0)
plt.show()

# 8. Distance range trend
plt.figure(figsize=(8, 5))
trend.plot(
    kind="line",
    marker="o"
)
plt.xlabel("Distance Range")
plt.ylabel("Average Delivery Time (minutes)")
plt.title("Delivery Time Trend by Distance")
plt.xticks(rotation=0)
plt.grid(True)
plt.show()

# 9. Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Between Logistics Variables")
plt.show()

print("\nAnalysis completed successfully.")
