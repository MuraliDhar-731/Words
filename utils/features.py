import textstat
import nltk
import joblib
import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

CSV_PATH = "data/dummy_dataset.csv"

def add_word_to_dataset(word, length, syllables, label):
    # Load existing or create new dataset
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        df = pd.DataFrame(columns=["word", "length", "syllables", "difficulty"])

    # Check if word already exists (optional)
    if word.lower() not in df["word"].str.lower().values:
        new_row = {"word": word, "length": length, "syllables": syllables, "difficulty": label}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)
        print(f"✅ Added: {word}")
    else:
        print(f"⚠️ Word already exists: {word}")

import pickle

def load_model():
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd

    data = {
        'length': [9, 4, 13, 10, 3],
        'syllables': [4, 2, 5, 5, 1],
        'difficulty': ['Medium', 'Easy', 'Hard', 'Medium', 'Easy']
    }

    df = pd.DataFrame(data)
    X = df[['length', 'syllables']]
    y = df['difficulty']

    model = RandomForestClassifier()
    model.fit(X, y)

    return model


nltk.download('averaged_perceptron_tagger')

def extract_features(word):
    return {
        'length': len(word),
        'syllables': textstat.syllable_count(word),
        'pos': nltk.pos_tag([word])[0][1]
    }


def predict_difficulty(model, features):
    vec = [features['length'], features['syllables']]
    return model.predict([vec])[0]
