import pandas as pd

data = pd.read_csv("f1_results.csv")

print("\n========== DATASET OVERVIEW ==========")
print("Shape:", data.shape)

print("\n========== COLUMNS ==========")
print(data.columns.tolist())

print("\n========== DATA TYPES ==========")
print(data.dtypes)

print("\n========== MISSING VALUES ==========")
print(data.isnull().sum())

print("\n========== SEASONS ==========")
print(sorted(data["season"].unique()))

print("\n========== RACE COUNT ==========")
print(data.groupby("season")["race"].nunique())

print("\n========== DRIVER COUNT ==========")
print(data["driver"].nunique())

print("\n========== CONSTRUCTOR COUNT ==========")
print(data["constructor"].nunique())

print("\n========== DUPLICATES ==========")
print(data.duplicated().sum())

print("\n========== FIRST 10 ROWS ==========")
print(data.head(10).to_string(index=False))



print("\n========== RECORDS PER RACE ==========")
race_counts = data.groupby(
    ["season", "round", "race"]
).size()

print(race_counts[race_counts != 20])

print("\n========== DRIVER COUNTS PER RACE ==========")
print(
    data.groupby(["season", "round"])["driver"]
    .nunique()
)

print("\n========== GRID POSITION RANGE ==========")
print(data["grid"].describe())

print("\n========== FINISHING POSITION RANGE ==========")
print(data["position"].describe())

print("\n========== UNIQUE FINISHING POSITIONS ==========")
print(sorted(data["position"].unique()))