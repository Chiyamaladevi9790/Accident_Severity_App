import pandas as pd
from catboost import CatBoostClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# ========== 1️⃣ Load Dataset ==========
try:
    df = pd.read_csv("RTA_Dataset.csv")
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    raise FileNotFoundError("❌ Error: 'RTA_Dataset.csv' not found. Please ensure it's in the same folder.")

# ========== 2️⃣ Basic Cleaning ==========
df = df.replace(["?", "na", "NA", "N/A", "--", " "], np.nan)

# ========== 3️⃣ Feature Engineering ==========
# Extract hour safely from "Time" column
if "Time" in df.columns:
    df["Hour"] = pd.to_datetime(df["Time"], errors="coerce").dt.hour.fillna(0).astype(int)

# Extract date-related features if date column exists
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date"].dt.year.fillna(0).astype(int)
    df["Month"] = df["Date"].dt.month.fillna(0).astype(int)

# ========== 4️⃣ Target & Features ==========
TARGET = "Accident_severity"
if TARGET not in df.columns:
    raise ValueError(f"❌ Target column '{TARGET}' not found in dataset!")

X = df.drop(columns=[TARGET])
y = df[TARGET]

# ========== 5️⃣ Handle Missing & Categorical Data ==========
cat_cols = X.select_dtypes(include="object").columns.tolist()

# Convert categorical to string and fill NaN
for col in cat_cols:
    X[col] = X[col].astype(str).fillna("Unknown")

# Fill numeric NaN with 0
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
X[num_cols] = X[num_cols].fillna(0)

# ========== 6️⃣ Train CatBoost Model ==========
print("📊 Training CatBoost model...")

model = CatBoostClassifier(
    iterations=800,
    depth=8,
    learning_rate=0.1,
    loss_function="MultiClass",
    verbose=100,
    random_seed=42
)

model.fit(X, y, cat_features=cat_cols)

# Save model and metadata
joblib.dump((model, cat_cols, num_cols), "accident_model.pkl")
print("✅ Model trained and saved as accident_model.pkl")

# ========== 7️⃣ Create EDA Charts Automatically ==========
if not os.path.exists("charts"):
    os.makedirs("charts")

# Chart 1: Severity distribution
plt.figure(figsize=(8, 5))
sns.countplot(x=TARGET, data=df, palette="coolwarm")
plt.title("Accident Severity Distribution")
plt.tight_layout()
plt.savefig("charts/severity_distribution.png")
plt.close()

# Chart 2: Severity by weather
if "Weather_conditions" in df.columns:
    plt.figure(figsize=(10, 5))
    sns.countplot(x="Weather_conditions", hue=TARGET, data=df, palette="mako")
    plt.title("Severity by Weather Conditions")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("charts/severity_by_weather.png")
    plt.close()

# Chart 3: Severity by hour
if "Hour" in df.columns:
    plt.figure(figsize=(10, 5))
    df.groupby("Hour")[TARGET].value_counts().unstack().plot(kind="bar", stacked=True, colormap="viridis")
    plt.title("Severity by Hour of Day")
    plt.xlabel("Hour (0–23)")
    plt.ylabel("Number of Accidents")
    plt.tight_layout()
    plt.savefig("charts/severity_by_hour.png")
    plt.close()

# Chart 4: Severity by Day of Week
if "Day_of_week" in df.columns:
    plt.figure(figsize=(10, 5))
    sns.countplot(x="Day_of_week", hue=TARGET, data=df, palette="plasma")
    plt.title("Severity by Day of Week")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("charts/severity_by_day.png")
    plt.close()

# Chart 5: Severity by Type of Vehicle
if "Type_of_vehicle" in df.columns:
    plt.figure(figsize=(10, 5))
    sns.countplot(x="Type_of_vehicle", hue=TARGET, data=df, palette="cubehelix")
    plt.title("Severity by Vehicle Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("charts/severity_by_vehicle.png")
    plt.close()

print("📈 All EDA charts saved in the /charts folder.")
print("🎯 Training completed successfully!")
