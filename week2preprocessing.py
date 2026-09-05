import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("zomato_cleaned.csv")

print("Original Shape:", df.shape)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Convert Excel time values into normal time format
def convert_time(x):
    if pd.isna(x):
        return x

    x = str(x)

    try:
        value = float(x)

        if 0 <= value <= 1:
            return (
                pd.Timestamp("1899-12-30") +
                pd.to_timedelta(value, unit="D")
            ).strftime("%H:%M")

    except:
        pass

    return x

df["Time_Orderd"] = df["Time_Orderd"].apply(convert_time)

# Fill missing values
df["Time_Orderd"] = df["Time_Orderd"].fillna("17:55")
df["Delivery_person_Age"] = df["Delivery_person_Age"].fillna(30)
df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].fillna(4.7)

# Check duplicate rows
print("\nDuplicate Rows:", df.duplicated().sum())

# Check for outliers using IQR
def check_outliers(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower) | (df[column] > upper)]

    print("\n", column)
    print("Lower Limit:", lower)
    print("Upper Limit:", upper)
    print("Outliers:", len(outliers))

check_outliers("Time_taken (min)")
check_outliers("distance_km")

# Normalize numerical columns
features = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Vehicle_condition",
    "multiple_deliveries",
    "Time_taken (min)",
    "distance_km"
]

scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Final check
print("\nFinal Missing Values:", df.isnull().sum().sum())
print("Final Duplicate Rows:", df.duplicated().sum())
print("Final Shape:", df.shape)

# Save the cleaned dataset
df.to_csv("zomato_preprocessed.csv", index=False)

print("\nPreprocessed dataset saved successfully.")
