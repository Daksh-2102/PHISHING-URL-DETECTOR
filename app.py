import streamlit as st
import pandas as pd
import re
from urllib.parse import urlparse
import joblib

# -------------------------
# Load trained model and feature list
# -------------------------
data = joblib.load("model_output/phishing_model_demo.joblib")
clf = data["model"]             # actual trained model
expected_features = data["features"]  # list of features used during training

# -------------------------
# Suspicious words list
# -------------------------
suspicious_words = [
    "secure", "account", "update", "login", "verify", "confirm",
    "signin", "bank", "paypal", "auth"
]

# -------------------------
# Feature extraction
# -------------------------
def extract_features(url):
    parsed = urlparse(url)
    host = parsed.netloc

    features = {
        "has_https": 1 if parsed.scheme == "https" else 0,
        "host_length": len(host),
        "num_digits": sum(c.isdigit() for c in url),
        "num_dots": url.count("."),
        "num_hyphen": url.count("-"),
        "length": len(url),
        "suspicious_word": int(any(word in url.lower() for word in suspicious_words)),
        "num_at": url.count("@"),
        "num_question": url.count("?"),
        "num_equal": url.count("="),
        "num_percent": url.count("%"),
        "num_slash": url.count("/"),
        "num_subdomain": host.count(".") - 1 if host.count(".") > 1 else 0
    }

    return pd.DataFrame([features])

# -------------------------
# Prediction function
# -------------------------
def predict_url(url):
    df_feats = extract_features(url)

    # Add missing features with default 0
    for feat in expected_features:
        if feat not in df_feats.columns:
            df_feats[feat] = 0

    # Reorder columns exactly as model expects
    df_feats = df_feats[expected_features]

    # Predict
    pred = clf.predict(df_feats)[0]
    # Optional: get probability of being phishing
    prob = clf.predict_proba(df_feats)[0][1]  # probability of phishing class
    return pred, prob

# -------------------------
# Streamlit app UI
# -------------------------
st.title("Phishing URL Detector")

user_url = st.text_input("Enter URL to check:", "")

if st.button("Check URL"):
    if user_url:
        try:
            result, probability = predict_url(user_url)
            if result == 1:
                st.error(f"⚠️ Phishing URL detected! Probability: {probability:.2f}")
            else:
                st.success(f"✅ URL looks safe. Probability of phishing: {probability:.2f}")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a URL.")
