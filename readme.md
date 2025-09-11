# Phishing URL Detector

## Overview
A Python-based ML project that classifies URLs as phishing or legitimate using RandomForest.

## Features
- Trains on Kaggle phishing dataset (~11,000 URLs)
- Extracts features: URL length, dots, hyphens, suspicious words, IP presence, HTTPS
- Saves trained model for predicting new URLs
- Outputs prediction with confidence score

## Usage
1. Install dependencies: pip install -r requirements.txt
or install all the dependcies i mentioned in requirements.txt file

2. Run the script:
3. Check terminal output for accuracy and example predictions.

## Demo
Example predictions:
- `http://192.168.1.5/secure-login` → phishing
- `https://www.google.com/search?q=test` → legitimate

##updation
##used streamlit to deploy my project

# Phishing URL Detector - Streamlit Web App

This project detects whether a URL is phishing or legitimate using a RandomForest ML model.  
It provides a **web interface** built with Streamlit.

## Features
- Input URL and get instant prediction.
- Confidence score shown.
- Feature values displayed for transparency.
- Example URLs provided for quick testing.

## How to Run
1. Clone the repo:
```bash
https://github.com/Daksh-2102/PHISHING-URL-DETECTOR

