import streamlit as st
import pickle
from textblob import TextBlob

model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))


st.title("📦 Amazon Product Review Sentiment Analysis")
st.write("Enter an Amazon review and find out if it's Positive, Negative, or Neutral.")

user_input = st.text_area("Enter your review here:")

if st.button("Analyze"):
    if user_input:
        # Transform input
        transformed = tfidf.transform([user_input])
        prediction = model.predict(transformed)[0]

        st.subheader(f"Predicted Sentiment: {prediction}")
        
        blob = TextBlob(user_input)
        st.write(f"Polarity Score: {blob.sentiment.polarity:.2f}")
    else:
        st.warning("Please enter a review first.")
