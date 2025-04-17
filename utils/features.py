import textstat
import nltk
import joblib
import os

nltk.download('averaged_perceptron_tagger')

def extract_features(word):
    return {
        'length': len(word),
        'syllables': textstat.syllable_count(word),
        'pos': nltk.pos_tag([word])[0][1]
    }

def load_model():
    return joblib.load("model/difficulty_model.pkl")

def predict_difficulty(model, features):
    vec = [features['length'], features['syllables']]
    return model.predict([vec])[0]
