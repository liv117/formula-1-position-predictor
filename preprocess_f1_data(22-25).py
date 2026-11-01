import pandas as pd

# Load the original dataset
data = pd.read_csv("f1_results_2022_2025_raw.csv")

print("Original dataset:", data.shape)

# Identify drivers who did not start
non_starters = ["Withdrew", "Did not start"]

# Create a separate training dataset
clean_data = data[
    ~data["status"].isin(non_starters)
].copy()

# Create an indicator for unusual grid positions
clean_data["grid_is_zero"] = (
    clean_data["grid"] == 0
).astype(int)

# Check the result
print("Cleaned dataset:", clean_data.shape)

print("\nRemaining statuses:")
print(clean_data["status"].value_counts())

print("\nGrid zero records:")
print(clean_data["grid_is_zero"].sum())

print("\nMissing values:")
print(clean_data.isnull().sum())



# Save cleaned dataset
clean_data.to_csv(
    "f1_results_2022_2025_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")