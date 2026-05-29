import pandas as pd
import random
import joblib

from sklearn.ensemble import RandomForestRegressor

cars = pd.read_csv("cars.csv")

training_data = []

# -----------------------------
# GENERATE TRAINING DATA
# -----------------------------

for _ in range(3000):

    budget = random.randint(20000, 120000)
    preferred_mpg = random.randint(20, 120)
    performance_interest = random.randint(1, 10)
    seating_amount = random.randint(2, 8)
    luxury_preference = random.randint(1, 10)
    reliability_importance = random.randint(1, 10)

    for _, car in cars.iterrows():

        score = 100

        score -= abs(budget - car["price"]) / 5000

        score -= abs(preferred_mpg - car["mpg"]) * 0.4

        score -= abs(
            performance_interest * 40 - car["horsepower"]
        ) / 30

        score -= abs(
            seating_amount - car["seats"]
        ) * 3

        score -= abs(
            luxury_preference - car["luxury"]
        ) * 2

        score -= abs(
            reliability_importance - car["reliability"]
        ) * 2

        score = max(50, min(100, score))

        training_data.append([
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
            car["reliability"],
            score
        ])

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

X = df.drop("score", axis=1)
y = df["score"]

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

joblib.dump(model, "model.pkl")

print("AI model trained successfully.")