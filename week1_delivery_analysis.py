import pandas as pd #sample data
data = {
    "Order_ID": [101, 102, 103, 104],
    "Expected_Hours": [24, 30, 20, 36],
    "Actual_Hours": [22, 35, 28, 40],
    "Distance_km": [5, 12, 8, 20],
    "Traffic": ["Low", "High", "Medium", "High"],
    "Weather": ["Clear", "Rain", "Clear", "Storm"]
}

df = pd.DataFrame(data)
df["Delay_Hours"] = df["Actual_Hours"] - df["Expected_Hours"]
df["Delayed"] = df["Delay_Hours"] > 0
print(df)
print("\nNumber of delayed orders:")
print(df["Delayed"].sum())
