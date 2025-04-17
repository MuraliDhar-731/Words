import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_difficulty_heatmap(words, difficulties):
    scores = [0 if d=='Easy' else 1 if d=='Medium' else 2 for d in difficulties]
    fig, ax = plt.subplots(figsize=(15, 1))
    sns.heatmap([scores], cmap="coolwarm", xticklabels=words, yticklabels=False, cbar=True, ax=ax)
    st.pyplot(fig)
