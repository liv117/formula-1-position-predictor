
import pandas as pd

# Load the feature-engineered dataset
data = pd.read_csv("f1_driver_features.csv")

# Display all unique circuits
circuits = (
    data[["circuit_id", "circuit", "country"]]
    .drop_duplicates()
    .sort_values("circuit")
    .reset_index(drop=True)
)

print("========== F1 CIRCUITS ==========")
print("Total unique circuits:", len(circuits))
print(circuits.to_string(index=False))
