
import pandas as pd

# Load cleaned dataset
data = pd.read_csv("f1_results_2022_2025_cleaned.csv")

# Convert race date
data["race_date"] = pd.to_datetime(data["race_date"])

# Create unique race ID
data["race_id"] = (
    data["season"].astype(str)
    + "_"
    + data["round"].astype(str)
)

# Get unique races in chronological order
races = (
    data[["race_id", "season", "round", "race_date", "race"]]
    .drop_duplicates()
    .sort_values(["race_date", "season", "round"])
    .reset_index(drop=True)
)

# Split races: 80% training, 20% testing
split_index = int(len(races) * 0.8)

train_races = races.iloc[:split_index]
test_races = races.iloc[split_index:]

# Get race IDs
train_ids = set(train_races["race_id"])
test_ids = set(test_races["race_id"])

# Split driver records by race
train_data = data[data["race_id"].isin(train_ids)].copy()
test_data = data[data["race_id"].isin(test_ids)].copy()

# Display results
print("========== TIME-BASED SPLIT ==========")

print("Total races:", len(races))
print("Training races:", len(train_races))
print("Testing races:", len(test_races))

print("Training records:", len(train_data))
print("Testing records:", len(test_data))

print("\nTraining period:")
print(train_races["race_date"].min(), "to",
      train_races["race_date"].max())

print("\nTesting period:")
print(test_races["race_date"].min(), "to",
      test_races["race_date"].max())

# Check race overlap
overlap = train_ids.intersection(test_ids)

print("\nRaces appearing in both sets:", len(overlap))

# Check chronological order
print(
    "Chronological split valid:",
    train_races["race_date"].max()
    < test_races["race_date"].min()
)
