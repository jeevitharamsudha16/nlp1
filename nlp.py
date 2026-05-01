import streamlit as st
import pickle
import re
import tensorflow as tf
import os

# use tensorflow namespace
load_model = tf.keras.models.load_model
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

# Optional: suppress TensorFlow logs

st.title("🎬 IMDB Sentiment Analysis")

# Load model & tokenizer
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("lstm_model.h5", compile=False)
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_resources()

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Prediction
def predict(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        seq, maxlen=100, padding="post"
    )
    prob = model.predict(padded)[0][0]
    return prob

# UI
review = st.text_area("Enter your movie review")

if st.button("Predict"):
    if review.strip() == "":
        st.warning("Please enter some text")
    else:
        prob = predict(review)

        st.write(f"Confidence: **{prob:.2f}**")
        st.progress(int(prob * 100))

        if prob >= 0.5:
            st.success("Positive 😊")
        else:
            st.error("Negative 😞")