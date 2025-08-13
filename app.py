import streamlit as st
import pickle
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import string
ps =PorterStemmer()

def transform_text(text):
    # for making text in lowercase:
    text = text.lower()
    # tokenaztion
    text = nltk.word_tokenize(text)
    y = []
    # Removing special Characters
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    # Removing Stop words and Punctuation

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    # Stemming:
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)



tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms=st.text_input("Enter the message")

if st.button("Predict"):
    #1. Preprocess
    transformed_sms = transform_text(input_sms)

    #2. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    #3.Predict
    result = model.predict(vector_input)[0]
    #4.Display
    if result == 1:
        st.header("Spam Alert!")
    else:
        st.header("Not Spam !")

