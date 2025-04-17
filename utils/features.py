import textstat
import nltk
import joblib
import os


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
