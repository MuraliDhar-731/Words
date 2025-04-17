import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("data/dummy_dataset.csv")
X = df[['length', 'syllables']]
y = df['difficulty']

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "model/difficulty_model.pkl")
print("✅ Model trained and saved!")
