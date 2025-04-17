import pandas as pd
import os
from datetime import datetime
import textstat
import nltk

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder

import streamlit as st

nltk.download('averaged_perceptron_tagger')

CSV_PATH = "data/dummy_dataset.csv"
HISTORY_LOG_PATH = "data/history.log"

# ✨ Extract features from a word
def extract_features(word):
    return {
        'length': len(word),
        'syllables': textstat.syllable_count(word),
        'pos': nltk.pos_tag([word])[0][1]
    }

# 🧠 Predict difficulty
def predict_difficulty(model, features):
    vec = [[features['length'], features['syllables']]]
    return model.predict(vec)[0]

# ➕ Add a new word to the dataset
def add_word_to_dataset(word, length, syllables, label):
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        df = pd.DataFrame(columns=["word", "length", "syllables", "difficulty"])

    if word.lower() not in df["word"].str.lower().values:
        new_row = {"word": word, "length": length, "syllables": syllables, "difficulty": label}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)
        log_retrain_event(word, label, len(df))

# 🕒 Log retrain events
def log_retrain_event(word, label, dataset_size):
    with open(HISTORY_LOG_PATH, "a") as log:
        log.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"Word added: '{word}' | Label: '{label}' | Dataset size: {dataset_size}\n")

# 🔁 Manual retraining (used by retrain button)
def retrain_model(fast_mode=False):
    df = pd.read_csv(CSV_PATH)
    if fast_mode:
        model = SGDClassifier(max_iter=1000, tol=1e-3)
    else:
        model = RandomForestClassifier()

    X = df[['length', 'syllables']]
    y = df['difficulty']
    model.fit(X, y)

    # Save model to cache (force update on retrain)
    st.cache_resource.clear()  # 💡 force reload next time
    return model

# ✅ Cached model loader
@st.cache_resource
def load_model_cached(fast_mode=False):
    df = pd.read_csv(CSV_PATH)
    if fast_mode:
        model = SGDClassifier(max_iter=1000, tol=1e-3)
    else:
        model = RandomForestClassifier()

    X = df[['length', 'syllables']]
    y = df['difficulty']
    model.fit(X, y)
    return model
