import streamlit as st
import pandas as pd
import numpy as np
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

tfidf = pickle.load(open('vectorizertfidf1.pkl', 'rb'))
model = pickle.load(open('modelmnb.pkl', 'rb'))

st.title("Spam Classifier")
input_text = st.text_input("Enter the text to be classified")

if st.button("Predict"):

    def lowering_text(text):
        text = text.lower()
        return text  

    def tokenizer(text):
        text = nltk.word_tokenize(text)
        return text

    def remove_special_characters(text):
        k = []
        for i in text:
            if i.isalnum():
                k.append(i)
        return k

    from nltk.stem.porter import PorterStemmer
    ps = PorterStemmer()

    def stemming(text):
        u=[]
        for i in text:
            u.append(ps.stem(i))
            
        return " ".join(u)


    a = lowering_text(input_text)
    b = tokenizer(a)
    c = remove_special_characters(b)
    d = stemming(c)

    tranformed_text = d

    tfidf_text = tfidf.transform([tranformed_text])

    result = model.predict(tfidf_text)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

