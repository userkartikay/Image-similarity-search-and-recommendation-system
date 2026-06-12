import pickle
import nltk
import streamlit as st
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string

nltk.download('all')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum() and i != "subject":
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

tfidf_two = pickle.load(open(r"C:\Users\karti\OneDrive\Desktop\projects\project1\vectorizer.pkl", 'rb'))
model_two = pickle.load(open(r"C:\Users\karti\OneDrive\Desktop\projects\project1\model.pkl", 'rb'))

st.title('spam ham classifier')
input_sms = st.text_input('Enter the message')
if st.button('predict'):
    transform_sms = transform_text(input_sms)
    vector_input = tfidf_two.transform([transform_sms])
    result = model_two.predict(vector_input)[0]
    if result == 'spam':
        st.header('Spam')
    else:
        st.header('not spam')