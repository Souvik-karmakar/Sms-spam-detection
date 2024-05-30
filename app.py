import streamlit as st
import pickle
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('punkt')

ps = PorterStemmer()

# Function to preprocess the input text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the TF-IDF vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess the input message
    transformed_sms = transform_text(input_sms)
    
    # 2. Vectorize the input message
    try:
        vector_input = tfidf.transform([transformed_sms])
    except Exception as e:
        st.error(f"Error in vectorizing input: {str(e)}")
        st.stop()
    
    # 3. Predict the class of the message
    try:
        result = model.predict(vector_input)[0]
    except Exception as e:
        st.error(f"Error in model prediction: {str(e)}")
        st.stop()
    
    # 4. Display the result
    if result == 1:
        st.header("Spam!")
    else:
        st.header("Not Spam!")
