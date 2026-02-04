import streamlit as st
import numpy as np
import re
import pickle
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# import nltk

@st.cache_resource
def download_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

download_nltk()


# Load saved assets
nb_model = joblib.load("naive_bayes_w2v.pkl")
rnn_model = load_model("rnn_sentiment_model.h5")
lstm_model = load_model("lstm_sentiment_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

w2v_model = Word2Vec.load("word2vec.model")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
max_len = 200

# ---------- Preprocessing ----------
def preprocess(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return tokens

def vectorize(tokens):
    vecs = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(100)

# ---------- Streamlit UI ----------
st.title("🎬 IMDb Sentiment Analysis")
st.write("Enter a movie review and see predictions from multiple models")

review = st.text_area("Type your movie review here")

if st.button("Analyze Sentiment"):

    tokens = preprocess(review)

    # Naive Bayes prediction
    vec = vectorize(tokens).reshape(1, -1)
    nb_pred = nb_model.predict(vec)[0]

    # RNN/LSTM prediction
    seq = tokenizer.texts_to_sequences([' '.join(tokens)])
    pad = pad_sequences(seq, maxlen=max_len, padding='post')

    rnn_pred = (rnn_model.predict(pad) > 0.5).astype(int)[0][0]
    lstm_pred = (lstm_model.predict(pad) > 0.5).astype(int)[0][0]

    def label(x): return "😊 Positive" if x == 1 else "😡 Negative"

    st.subheader("Results")
    st.write(f"**Naive Bayes:** {label(nb_pred)}")
    st.write(f"**RNN:** {label(rnn_pred)}")
    st.write(f"**LSTM:** {label(lstm_pred)}")
