"""
Script to train and save the ML model
Run this once to generate the pickled model
"""

import pickle
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Create directories if they don't exist
os.makedirs('model', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create a dataframe and save as CSV
df = pd.DataFrame(X, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df['target'] = y
df.to_csv('data/iris.csv', index=False)

print(f"Dataset saved: {len(df)} samples")

# Train a simple Random Forest model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"Model accuracy: {score:.4f}")

# Save the model
model_path = 'model/model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved to {model_path}")
