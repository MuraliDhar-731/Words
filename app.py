import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from textstat import syllable_count

from utils.features import (
    extract_features,
    predict_difficulty,
    load_model_cached,
    retrain_model,
    add_word_to_dataset,
)

from utils.extract_text import get_text_from_url, get_text_from_upload
from utils.definitions import get_word_info
from utils.heatmap import plot_difficulty_heatmap

nltk.download('punkt')

st.title("🧠 Word Difficulty Predictor")

# 🚀 Mode Toggle
mode = st.radio("Choose Model Mode:", ["⚡ Light Mode (Fast)", "🎯 Full Mode (Accurate)"])
fast_mode = mode == "⚡ Light Mode (Fast)"

# ✅ Cached Model Load
model = load_model_cached(fast_mode)

# 📥 Input Section
col1, col2 = st.columns(2)
text = ""

with col1:
    url = st.text_input("🔗 Enter URL")
    if st.button("Fetch"):
        text = get_text_from_url(url)

with col2:
    uploaded = st.file_uploader("📄 Upload .txt file", type=["txt"])
    if uploaded:
        text = get_text_from_upload(uploaded)

if not text:
    text = st.text_area("Or paste paragraph here:")

# 📊 Word Difficulty Prediction
if text:
    st.subheader("📊 Predictions")
    words = word_tokenize(text)
    filtered_words = [w for w in words if w.isalpha()]
    difficulties, table = [], []

    for word in filtered_words[:100]:
        features = extract_features(word)
        label = predict_difficulty(model, features)
        meaning, example = get_word_info(word)
        table.append((word, label, meaning, example))
        difficulties.append(label)

    for word, label, meaning, example in table:
        st.markdown(f"**{word}** → `{label}`")
        st.markdown(f"📘 *Definition*: {meaning}")
        st.markdown(f"✏️ *Example*: {example[:1] if example else 'No example'}")
        st.markdown("---")

    st.subheader("🔥 Heatmap")
    plot_difficulty_heatmap(filtered_words[:len(difficulties)], difficulties)

# ➕ Add New Word
with st.expander("➕ Add a new word manually"):
    new_word = st.text_input("Enter word:")
    new_label = st.selectbox("Select difficulty label", ["Easy", "Medium", "Hard"])
    if st.button("Add & Save"):
        length = len(new_word)
        syllables = syllable_count(new_word)
        add_word_to_dataset(new_word, length, syllables, new_label)
        st.success(f"'{new_word}' added. You can now retrain manually.")

# 🔁 Retrain Model
if st.button("🔁 Retrain Model Now"):
    retrain_model(fast_mode)
    st.success("Model retrained!")

# 🕒 View History Log
with st.expander("🕒 Retrain History Log"):
    search_query = st.text_input("🔍 Search retrain history (e.g., 'Hard', 'education', '2025')")
    try:
        with open("data/history.log", "r") as f:
            logs = f.readlines()
        filtered = [line for line in logs if search_query.lower() in line.lower()] if search_query else logs[-50:]
        if filtered:
            st.text("".join(filtered))
        else:
            st.warning("🚫 No matching entries found.")
    except FileNotFoundError:
        st.info("No retraining history found yet.")

# 📂 Search + Predict from Dataset
with st.expander("📂 View Predictions for Dataset Words"):
    search_dataset = st.text_input("🔎 Search word in dataset")
    try:
        df = pd.read_csv("data/dummy_dataset.csv")
        df["predicted"] = df.apply(
            lambda row: predict_difficulty(
                model, {"length": row["length"], "syllables": row["syllables"], "pos": "NN"}
            ),
            axis=1,
        )
        if search_dataset:
            df = df[df["word"].str.contains(search_dataset, case=False)]
        st.dataframe(df[["word", "length", "syllables", "difficulty", "predicted"]])
    except FileNotFoundError:
        st.warning("❗ Dataset not found.")
