import requests
import csv
import os

print("Current folder:", os.getcwd())

url = "https://api.jolpi.ca/ergast/f1/2025/1/results.json"

response = requests.get(url)

print(response.status_code)

data = response.json()

race = data["MRData"]["RaceTable"]["Races"][0]

race_name = race["raceName"]
race_date = race["date"]

print("Race:", race_name)
print("Date:", race_date)

results = race["Results"]

with open("f1_results.csv", "w", newline="", encoding="utf-8") as file:

    writer = csv.writer(file)

    writer.writerow([
        "season",
        "round",
        "race",
        "driver",
        "constructor",
        "grid",
        "points",
        "laps",
        "position"
    ])

    for result in results:

        writer.writerow([
            race["season"],
            race["round"],
            race["raceName"],
            result["Driver"]["givenName"] + " " + result["Driver"]["familyName"],
            result["Constructor"]["name"],
            result["grid"],
            result["points"],
            result["laps"],
            result["position"]
        ])

print("CSV file created successfully!")
