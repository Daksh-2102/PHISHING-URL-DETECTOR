"""
Phishing URL Detector
- Uses RandomForest ML model to classify URLs as phishing or legitimate.
- Features include URL length, dots, hyphens, IP, suspicious words, etc.
- Trained on Kaggle phishing dataset.
"""

import os
import re
from urllib.parse import urlparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# -----------------------------
# ---- Config & Paths ---------
# -----------------------------
dataset_path = os.path.join(os.getcwd(), "phishing.csv")
model_dir = os.path.join(os.getcwd(), "model_output")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "phishing_model_demo.joblib")

# -----------------------------
# ---- Load Dataset -----------
# -----------------------------
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please add phishing.csv")

df = pd.read_csv(dataset_path)
print("Dataset loaded:", df.shape)

# -----------------------------
# ---- Feature Extraction -----
# -----------------------------
suspicious_words = ["secure","account","update","login","verify","confirm","signin","bank","paypal","auth"]

def has_ip(address: str) -> int:
    """Check if URL contains an IP address"""
    return int(bool(re.search(r"(^|\.)\d{1,3}(\.\d{1,3}){3}($|:)", address)))

def extract_features(url: str) -> dict:
    """Extract simple URL-based features"""
    url = url.lower()
    host = urlparse(url).netloc
    path = urlparse(url).path
    suspicious_count = sum(1 for w in suspicious_words if w in url)
    
    return {
        "url_length": len(url),
        "num_dots": url.count("."),
        "num_hyphen": url.count("-"),
        "num_digits": sum(c.isdigit() for c in url),
        "has_https": int(url.startswith("https")),
        "has_ip": has_ip(host),
        "path_length": len(path),
        "suspicious_words_count": suspicious_count,
        "host_length": len(host)
    }

# -----------------------------
# ---- Prepare Training Data ---
# -----------------------------
X = pd.DataFrame([extract_features(u) for u in df["url"]])
y = df["status"].apply(lambda x: 1 if x.lower() == "phishing" else 0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y, shuffle=True
)

# -----------------------------
# ---- Train Model -------------
# -----------------------------
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# -----------------------------
# ---- Evaluate Model ----------
# -----------------------------
y_pred = clf.predict(X_test)
print("\nTest accuracy:", round(accuracy_score(y_test, y_pred), 3))
print(classification_report(y_test, y_pred, target_names=["legit","phishing"]))

# -----------------------------
# ---- Save Model --------------
# -----------------------------
joblib.dump({"model": clf, "features": list(X.columns)}, model_path)
print(f"Model saved at {model_path}")

# -----------------------------
# ---- Prediction Helper -------
# -----------------------------
def predict_url(url: str) -> dict:
    """Predict if URL is phishing or legit"""
    try:
        feats = extract_features(url)
        df_feats = pd.DataFrame([feats])
        pred = clf.predict(df_feats)[0]
        prob = clf.predict_proba(df_feats)[0][pred]
        return {
            "url": url,
            "prediction": "phishing" if pred == 1 else "legit",
            "probability": round(prob, 3),
            "features": feats
        }
    except Exception as e:
        return {"url": url, "error": str(e)}

# -----------------------------
# ---- Quick Demo -------------
# -----------------------------
if __name__ == "__main__":
    from pprint import pprint

    examples = [
        "http://192.168.1.5/secure-login",
        "https://www.paypal.com/signin",
        "http://free-money-bank.com/claim"
    ]

    print("\n--- Demo Predictions ---\n")
    for url in examples:
        pprint(predict_url(url))
