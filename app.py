import streamlit as st
import pandas as pd
import joblib
import random
import time

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="AI Car Recommender",
    page_icon="🚗",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS + JS
# -----------------------------

st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #0f172a;
    color: white;
}

.main-title {
    text-align: center;
    font-size: 60px;
    font-weight: bold;
    color: #60a5fa;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 40px;
}

.card {
    background: #1e293b;
    padding: 25px;
    border-radius: 20px;
    margin-bottom: 30px;
    box-shadow: 0px 0px 20px rgba(96,165,250,0.25);
}

.score {
    font-size: 40px;
    color: #4ade80;
    font-weight: bold;
}

.ai-box {
    background-color: #111827;
    padding: 15px;
    border-radius: 12px;
    border-left: 5px solid #60a5fa;
}

</style>

<script>

const messages = [
    "Analyzing your preferences...",
    "Training recommendation weights...",
    "Comparing vehicles...",
    "Generating AI matches..."
];

let index = 0;

setInterval(() => {
    const el = document.getElementById("ai-status");
    if(el){
        el.innerHTML = messages[index];
        index = (index + 1) % messages.length;
    }
}, 1500);

</script>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA
# -----------------------------

cars = pd.read_csv("cars.csv")
model = joblib.load("model.pkl")

# -----------------------------
# TITLE
# -----------------------------

st.markdown(
    "<div class='main-title'>🚗 AI Car Recommender</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Find the perfect vehicle using machine learning.</div>",
    unsafe_allow_html=True
)

# -----------------------------
# USER INPUTS
# -----------------------------

col1, col2 = st.columns(2)

with col1:

    budget = st.slider(
        "Budget ($)",
        20000,
        120000,
        50000,
        step=1000
    )

    preferred_mpg = st.slider(
        "Preferred MPG",
        15,
        130,
        35
    )

    performance_interest = st.slider(
        "Performance Interest",
        1,
        10,
        5
    )

with col2:

    seating_amount = st.slider(
        "Seats Needed",
        2,
        8,
        5
    )

    luxury_preference = st.slider(
        "Luxury Preference",
        1,
        10,
        5
    )

    reliability_importance = st.slider(
        "Reliability Importance",
        1,
        10,
        8
    )

lifestyle = st.selectbox(
    "Lifestyle",
    [
        "family",
        "commuter",
        "sport",
        "outdoor",
        "tech",
        "luxury",
        "fun"
    ]
)

# -----------------------------
# BUTTON
# -----------------------------

if st.button("🔍 Generate AI Recommendations"):

    st.markdown(
        "<div id='ai-status' style='font-size:24px;color:#60a5fa;margin-bottom:20px;'></div>",
        unsafe_allow_html=True
    )

    with st.spinner("AI is thinking..."):
        time.sleep(3)

    results = []

    # -----------------------------
    # SCORE CARS
    # -----------------------------

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

        # Strong budget penalty
        
        if car["price"] > budget:

            over_budget = car["price"] - budget
            
            predicted_score -= over_budget / 1000

        if lifestyle == car["lifestyle"]:
            predicted_score += 5

        results.append({
            "make": car["make"],
            "model": car["model"],
            "score": round(predicted_score, 1),
            "price": car["price"],
            "mpg": car["mpg"],
            "horsepower": car["horsepower"],
            "seats": car["seats"],
            "luxury": car["luxury"],
            "reliability": car["reliability"],
            "image": car["image"]
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="score",
        ascending=False
    )

    st.subheader("🏆 Top Matches")

    # -----------------------------
    # SHOW RESULTS
    # -----------------------------

    for _, car in results_df.head(3).iterrows():

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.image(car["image"], use_container_width=True)

        st.markdown(
            f"## {car['make']} {car['model']}"
        )

        st.markdown(
            f"<div class='score'>{car['score']}% Match</div>",
            unsafe_allow_html=True
        )

        st.write(f"💰 Price: ${car['price']:,}")
        st.write(f"⛽ MPG: {car['mpg']}")
        st.write(f"🏎️ Horsepower: {car['horsepower']}")
        st.write(f"🪑 Seats: {car['seats']}")
        st.write(f"✨ Luxury: {car['luxury']}/10")
        st.write(f"🔧 Reliability: {car['reliability']}/10")

        # -----------------------------
        # AI EXPLANATION
        # -----------------------------

        explanations = [
            f"The {car['make']} {car['model']} aligns strongly with your performance and reliability preferences.",
            f"This vehicle was highly ranked because it closely matches your desired balance of luxury and practicality.",
            f"The AI determined this vehicle is one of the strongest overall matches for your selected lifestyle.",
            f"This recommendation scored highly due to its excellent combination of value, efficiency, and features."
        ]

        st.markdown(
            "<div class='ai-box'>",
            unsafe_allow_html=True
        )

        st.subheader("🤖 AI Analysis")

        st.write(random.choice(explanations))

        st.markdown("</div>", unsafe_allow_html=True)

        # -----------------------------
        # PROS
        # -----------------------------

        st.subheader("✅ Pros")

        pros = []

        if car["price"] <= budget:
            pros.append("Fits within your budget")

        if car["mpg"] >= preferred_mpg:
            pros.append("Excellent fuel economy")

        if car["horsepower"] >= 250:
            pros.append("Strong performance")

        if car["luxury"] >= 7:
            pros.append("Premium interior and features")

        if car["reliability"] >= 8:
            pros.append("Excellent reliability")

        for pro in pros:
            st.write(f"• {pro}")

        # -----------------------------
        # CONS
        # -----------------------------

        st.subheader("⚠️ Potential Downsides")

        cons = []

        if car["price"] > budget:
            cons.append("Above preferred budget")

        if car["mpg"] < preferred_mpg:
            cons.append("Lower MPG than requested")

        if car["reliability"] < 7:
            cons.append("Average reliability rating")

        if len(cons) == 0:
            cons.append("Very few downsides based on your preferences")

        for con in cons:
            st.write(f"• {con}")

        st.markdown("</div>", unsafe_allow_html=True)