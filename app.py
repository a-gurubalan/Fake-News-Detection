import streamlit as st
import joblib

st.title("Fake News Detector")

try:
    vectorizer = joblib.load("vectorizer.jb")
    model = joblib.load("lr_model.jb")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

news_input = st.text_area("News Article:")

if st.button("Check News"):
    if news_input.strip():
        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)

        if prediction[0] == 1:
            st.success("Real News ✅")
        else:
            st.error("Fake News ❌")
    else:
        st.warning("Enter text")