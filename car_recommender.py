# ml_car_recommender.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# -----------------------------
# CAR DATABASE
# -----------------------------

cars_data = [
    {
        "make": "Toyota",
        "model": "Camry",
        "price": 30000,
        "mpg": 32,
        "horsepower": 203,
        "seats": 5,
        "luxury": 4,
        "reliability": 9,
        "lifestyle": "family"
    },
    {
        "make": "Honda",
        "model": "Civic",
        "price": 27000,
        "mpg": 36,
        "horsepower": 180,
        "seats": 5,
        "luxury": 3,
        "reliability": 9,
        "lifestyle": "commuter"
    },
    {
        "make": "BMW",
        "model": "M340i",
        "price": 62000,
        "mpg": 26,
        "horsepower": 382,
        "seats": 5,
        "luxury": 9,
        "reliability": 6,
        "lifestyle": "sport"
    },
    {
        "make": "Mazda",
        "model": "MX-5 Miata",
        "price": 32000,
        "mpg": 30,
        "horsepower": 181,
        "seats": 2,
        "luxury": 5,
        "reliability": 8,
        "lifestyle": "fun"
    },
    {
        "make": "Subaru",
        "model": "Outback",
        "price": 35000,
        "mpg": 29,
        "horsepower": 182,
        "seats": 5,
        "luxury": 5,
        "reliability": 8,
        "lifestyle": "outdoor"
    },
    {
        "make": "Tesla",
        "model": "Model 3",
        "price": 42000,
        "mpg": 120,
        "horsepower": 283,
        "seats": 5,
        "luxury": 8,
        "reliability": 7,
        "lifestyle": "tech"
    }
]

cars = pd.DataFrame(cars_data)

# -----------------------------
# TRAINING DATA
# -----------------------------
# These are EXAMPLE user preference patterns
# and how much they liked each car.
#
# In a real system this comes from:
# - user ratings
# - clicks
# - purchases
# - saved vehicles

training_data = [
    # budget, mpg_pref, perf, seats, luxury, reliability, car_price,
    # car_mpg, car_hp, car_seats, car_lux, car_rel, target_score

    [30000, 35, 4, 5, 3, 9, 27000, 36, 180, 5, 3, 9, 95],
    [50000, 25, 9, 5, 9, 5, 62000, 26, 382, 5, 9, 6, 91],
    [35000, 30, 5, 5, 5, 8, 35000, 29, 182, 5, 5, 8, 93],
    [45000, 100, 7, 5, 8, 7, 42000, 120, 283, 5, 8, 7, 96],
    [32000, 28, 8, 2, 5, 8, 32000, 30, 181, 2, 5, 8, 94],
    [28000, 40, 3, 5, 2, 10, 30000, 32, 203, 5, 4, 9, 90],
]

columns = [
    "budget",
    "preferred_mpg",
    "performance_interest",
    "seating_amount",
    "luxury_preference",
    "reliability_importance",
    "car_price",
    "car_mpg",
    "car_hp",
    "car_seats",
    "car_luxury",
    "car_reliability",
    "score"
]

df = pd.DataFrame(training_data, columns=columns)

# -----------------------------
# MACHINE LEARNING SETUP
# -----------------------------

X = df.drop("score", axis=1)
y = df["score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# USER INPUT
# -----------------------------

description = input(
    "Describe your ideal car: "
).lower()

# Default preferences
budget = 50000
preferred_mpg = 30
performance_interest = 5
seating_amount = 5
luxury_preference = 5
reliability_importance = 5

# Simple NLP-ish parsing

if "sport" in description or "fast" in description:
    performance_interest = 9

if "luxury" in description:
    luxury_preference = 9

if "reliable" in description:
    reliability_importance = 10

if "family" in description:
    seating_amount = 5

if "fuel efficient" in description or "good mpg" in description:
    preferred_mpg = 40

# Budget extraction

words = description.split()

for word in words:

    if "k" in word:

        try:
            budget = int(word.replace("k", "")) * 1000
        except:
            pass

# -----------------------------
# PREDICT SCORES
# -----------------------------

results = []

for _, car in cars.iterrows():

    features = [[
        budget,
        preferred_mpg,
        performance_interest,
        seating_amount,
        luxury_preference,
        reliability_importance,
        car["price"],
        car["mpg"],
        car["horsepower"],
        car["seats"],
        car["luxury"],
        car["reliability"]
    ]]

    predicted_score = model.predict(features)[0]

    results.append({
        "make": car["make"],
        "model": car["model"],
        "predicted_score": round(predicted_score, 2),
        "price": car["price"]
    })

# -----------------------------
# RESULTS
# -----------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="predicted_score",
    ascending=False
)

print("\n=== ML CAR RECOMMENDATIONS ===\n")

id="f2f9xf"
for _, result in results_df.head(3).iterrows():

    # Find original car data
    original_car = cars[
        (cars["make"] == result["make"]) &
        (cars["model"] == result["model"])
    ].iloc[0]

    print(f"\n{result['make']} {result['model']}")
    print("=" * 40)

    print(f"Predicted Match Score: {result['predicted_score']}")

    # -----------------------------
    # SPECS
    # -----------------------------

    print("\nSPECS")
    print(f"Price: ${original_car['price']}")
    print(f"MPG: {original_car['mpg']}")
    print(f"Horsepower: {original_car['horsepower']}")
    print(f"Seats: {original_car['seats']}")
    print(f"Luxury Rating: {original_car['luxury']}/10")
    print(f"Reliability: {original_car['reliability']}/10")
    print(f"Lifestyle Category: {original_car['lifestyle']}")

    # -----------------------------
    # WHY IT MATCHES
    # -----------------------------

    reasons = []

    if original_car["price"] <= budget:
        reasons.append("Fits within your budget")

    if original_car["mpg"] >= preferred_mpg:
        reasons.append("Matches your fuel economy needs")

    if (
        performance_interest >= 7
        and original_car["horsepower"] >= 250
    ):
        reasons.append("Strong performance and horsepower")

    if (
        luxury_preference >= 7
        and original_car["luxury"] >= 7
    ):
        reasons.append("Luxury features match your preference")

    if original_car["reliability"] >= 8:
        reasons.append("Known for high reliability")

    if original_car["seats"] >= seating_amount:
        reasons.append("Enough seating for your needs")

    # Print explanations
    print("\nWHY IT MATCHES")

    for reason in reasons:
        print(f"- {reason}")

    # Possible downsides
    downsides = []

    if original_car["price"] > budget:
        downsides.append("Above your budget")

    if original_car["mpg"] < preferred_mpg:
        downsides.append("Lower MPG than requested")

    if original_car["seats"] < seating_amount:
        downsides.append("May not have enough seating")

    if original_car["reliability"] < 7:
        downsides.append("Average reliability")

    print("\nPOTENTIAL DOWNSIDES")

    if downsides:
        for downside in downsides:
            print(f"- {downside}")
    else:
        print("- Very few downsides based on your preferences")

    print("\n" + "-" * 50)