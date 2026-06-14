# check_features.py

import joblib

features = joblib.load("../models/feature_names.pkl")

for feature in features:
    print(feature)