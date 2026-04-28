import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="AI USA Housing Predictor", layout="wide")

# ----------------------------
# FULL THEME
# ----------------------------
st.markdown("""
<style>

/* Main background (sky → grey → grass) */
.stApp {
    background: linear-gradient(
        to bottom,
        #87CEEB 0%,     /* sky blue */
        #e0e0e0 50%,    /* grey middle */
        #4caf50 100%    /* grass */
    );
}

/* Top header bar */
header[data-testid="stHeader"] {
    background-color: #8B0000 !important;
    box-shadow: none;
}

header[data-testid="stHeader"] * {
    color: white !important;
}

/* ----------------------------
   SCROLLBAR (LIGHT BLUE)
---------------------------- */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: #cfd8dc;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #b71c1c, #8B0000);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #ff5252, #c62828);
}

/* Firefox */
* {
    scrollbar-width: thin;
    scrollbar-color: #8B0000 #cfd8dc;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD MODEL
# ----------------------------
model = pickle.load(open("model.pkl", "rb"))

# ----------------------------
# SIDEBAR INFO PANEL
# ----------------------------
with st.sidebar:
    st.markdown("## 🏠 Model Info")

    st.markdown("### Model Type")
    st.write("Random Forest Regressor")

    st.markdown("### Features Used")
    st.write("""
- Avg Area Income  
- House Age  
- Avg Number of Rooms  
- Avg Number of Bedrooms  
- Area Population
    """)

    st.markdown("### Description")
    st.write("Predicts U.S. housing prices using machine learning regression.")

# ----------------------------
# TITLE
# ----------------------------
st.title("🏠 USA Housing Price Predictor")

st.write("Predict housing prices using U.S. housing dataset features.")

# ----------------------------
# INPUTS
# ----------------------------
income = st.number_input("Avg Area Income", 20000.0, 150000.0, 70000.0, step=100.0)
age = st.number_input("Avg Area House Age", 1.0, 20.0, 5.0, step=0.1)
rooms = st.number_input("Avg Area Number of Rooms", 1.0, 10.0, 6.0, step=0.1)
bedrooms = st.number_input("Avg Area Number of Bedrooms", 1.0, 10.0, 3.0, step=0.1)
population = st.number_input("Area Population", 1000.0, 100000.0, 30000.0, step=100.0)

# ----------------------------
# PREDICTION
# ----------------------------
if st.button("Predict Price"):

    features = pd.DataFrame([{
        "Avg. Area Income": income,
        "Avg. Area House Age": age,
        "Avg. Area Number of Rooms": rooms,
        "Avg. Area Number of Bedrooms": bedrooms,
        "Area Population": population
    }])

    prediction = model.predict(features)[0]

    formatted_price = f"{prediction:,.2f}"

    st.subheader(f"💰 Estimated Price: ${formatted_price}")

    # ----------------------------
    # CONFIDENCE SCORE
    # ----------------------------
    tree_preds = np.array([
        tree.predict(features)[0]
        for tree in model.estimators_
    ])

    std = np.std(tree_preds)
    confidence = max(0, 100 - (std / (prediction + 1)) * 100)

    st.write(f"🔍 Confidence Score: {confidence:.2f}%")

    # Explanation
    st.markdown("""
### 📊 Why this confidence score?
This score is based on how much the individual decision trees in the Random Forest disagree.

- If all trees give similar predictions → high confidence  
- If predictions vary widely → lower confidence  

### ⚠️ Disclaimer
This is an estimation model. Real housing prices can vary due to:
- location differences
- market conditions
- property quality details not in dataset
""")