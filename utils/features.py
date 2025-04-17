import textstat
import nltk
import joblib
import os


import pickle

def load_model():
    with open("model/difficulty_model_pickle.pkl", "rb") as f:
        return pickle.load(f)

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
