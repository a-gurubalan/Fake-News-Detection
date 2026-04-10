import streamlit as st
import joblib

# 🔥 ADD LOGO HERE
st.image("logo.png", width=250)
st.title("Fake News Detector")

try:
    vectorizer = joblib.load("vectorizer.jb")
    model = joblib.load("lr_model.jb")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

news_input = st.text_area("News Article:")

if st.button("check News"):
    if news_input.strip():
        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)

        proba = model.predict_proba(transform_input)
        real_prob = proba[0][1] * 100
        fake_prob = proba[0][0] * 100

        if prediction[0] == 1:
            st.success("The News is Real! ✅")
        else:
            st.error("The News is Fake! ❌")

        st.write(f"Real: {real_prob:.2f}%")
        st.write(f"Fake: {fake_prob:.2f}%")

    else:
        st.warning("Please enter some text")
