# train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# --- Model Training ---

# Load the dataset
# Make sure 'crop_recommendation.csv' is in the same folder as this script
try:
    df = pd.read_csv(r"C:\Users\nmvij\Downloads\crop_recomendation.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'crop_recommendation.csv' not found. Please place it in the project folder.")
    exit()

# Separate features (X) and target (y)
X = df.drop('label', axis=1)
y = df['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(random_state=42)
print("Training the model...")
model.fit(X_train, y_train)
print("Model training complete.")

# --- Save the Trained Model to a File ---
# We are saving the trained model so our web app can use it later
model_filename = 'crop_recommendation.model'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Model saved successfully as '{model_filename}'")