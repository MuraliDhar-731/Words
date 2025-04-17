import streamlit as st
from utils.extract_text import get_text_from_url, get_text_from_upload
from utils.features import extract_features, predict_difficulty, load_model
from utils.definitions import get_word_info
from utils.heatmap import plot_difficulty_heatmap
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

st.set_page_config(page_title="Word Difficulty Predictor", layout="wide")
st.title("🧠 Word Difficulty Predictor")
st.write("Upload text or URL and get word difficulty, definitions, examples, and heatmap.")

model = load_model()

col1, col2 = st.columns(2)
text = ""

with col1:
    url = st.text_input("🔗 Enter URL to fetch text:")
    if st.button("Fetch from URL"):
        text = get_text_from_url(url)

with col2:
    uploaded = st.file_uploader("📄 Upload a .txt file", type=["txt"])
    if uploaded:
        text = get_text_from_upload(uploaded)

if not text:
    text = st.text_area("📝 Or paste your own paragraph here:")

if text:
    st.subheader("📊 Word Predictions")
    words = word_tokenize(text)
    filtered_words = [w for w in words if w.isalpha()]
    difficulties = []
    table = []

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



with st.expander("➕ Add a new word manually"):
    new_word = st.text_input("Enter word:")
    new_label = st.selectbox("Select difficulty label", ["Easy", "Medium", "Hard"])
    if st.button("Add & Retrain"):
        from textstat import syllable_count
        from utils.features import add_word_to_dataset, load_model

        length = len(new_word)
        syllables = syllable_count(new_word)
        add_word_to_dataset(new_word, length, syllables, new_label)
        model = load_model()
        st.success(f"'{new_word}' added and model retrained!")
with st.expander("🕒 Retrain History Log"):
    try:
        with open("data/history.log", "r") as f:
            logs = f.read()
        st.text(logs)
    except FileNotFoundError:
        st.info("No retraining history found yet.")
