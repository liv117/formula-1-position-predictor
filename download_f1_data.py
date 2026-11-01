
import requests
import pandas as pd
import time

# 1. CONFIGURATION

SEASONS = [2022, 2023, 2024, 2025]

BASE_URL = "https://api.jolpi.ca/ergast/f1"

HEADERS = {
    "User-Agent": "F1PositionPredictor/1.0"
}

all_results = []


# 2. DOWNLOAD EACH SEASON

for season in SEASONS:

    print(f"\nDownloading {season} season...")

    url = f"{BASE_URL}/{season}/results.json"

    offset = 0
    limit = 100

    # Store race information and combine pages
    races_by_key = {}

    while True:

        params = {
            "limit": limit,
            "offset": offset
        }

        response = requests.get(
            url,
            params=params,
            headers=HEADERS,
            timeout=30
        )

        response.raise_for_status()

        api_data = response.json()["MRData"]

        races = api_data["RaceTable"].get("Races", [])
        total = int(api_data["total"])

        page_result_count = 0

        for race in races:

            key = (
                race["season"],
                race["round"]
            )

            if key not in races_by_key:

                race_info = {
                    k: v for k, v in race.items()
                    if k != "Results"
                }

                race_info["Results"] = []

                races_by_key[key] = race_info

            results = race.get("Results", [])

            races_by_key[key]["Results"].extend(results)

            page_result_count += len(results)

        print(
            f"Retrieved {offset + page_result_count} "
            f"of {total} driver records"
        )

        if page_result_count == 0:
            break

        offset += page_result_count

        if offset >= total:
            break

        time.sleep(0.3)

    # 3. EXTRACT DRIVER RESULTS

    for race in races_by_key.values():

        circuit = race.get("Circuit") or {}
        location = circuit.get("Location") or {}

        for result in race.get("Results", []):

            driver = result.get("Driver") or {}
            constructor = result.get("Constructor") or {}

            row = {
                "season": int(race["season"]),
                "round": int(race["round"]),
                "race": race["raceName"],
                "race_date": race.get("date"),

                "circuit_id": circuit.get("circuitId"),
                "circuit": circuit.get("circuitName"),
                "country": location.get("country"),
                "locality": location.get("locality"),
                "latitude": location.get("lat"),
                "longitude": location.get("long"),

                "driver_id": driver.get("driverId"),
                "driver": (
                    driver.get("givenName", "")
                    + " "
                    + driver.get("familyName", "")
                ).strip(),

                "constructor_id": constructor.get("constructorId"),
                "constructor": constructor.get("name"),

                "grid": result.get("grid"),
                "position": result.get("position"),
                "position_text": result.get("positionText"),
                "points": result.get("points"),
                "laps": result.get("laps"),
                "status": result.get("status")
            }

            all_results.append(row)

    print(
        f"Completed {season}: "
        f"{len(races_by_key)} races"
    )


# 4. CREATE DATAFRAME

data = pd.DataFrame(all_results)

numeric_columns = [
    "season", "round", "grid", "position",
    "points", "laps", "latitude", "longitude"
]

for column in numeric_columns:

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )

data = data.sort_values(
    ["season", "round", "position"],
    na_position="last"
).reset_index(drop=True)


# 5. SAVE NEW DATASET

output_file = "f1_results_2022_2025_raw.csv"

data.to_csv(output_file, index=False)


# 6. DOWNLOAD SUMMARY

print("\n========== DOWNLOAD COMPLETE ==========")

print("Dataset shape:", data.shape)

print("\nRaces per season:")
print(
    data.groupby("season")["round"].nunique()
)

print("\nDriver records per season:")
print(
    data.groupby("season").size()
)

print("\nMissing values:")
print(data.isnull().sum())

print("\nDuplicate driver-race records:")
print(
    data.duplicated(
        subset=["season", "round", "driver_id"]
    ).sum()
)

print("\nMissing numeric finishing positions:")
print(data["position"].isna().sum())

print(f"\nSaved as: {output_file}")
