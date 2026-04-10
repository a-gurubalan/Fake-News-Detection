import streamlit as st
import joblib

st.set_page_config(page_title="Fake News Detector", layout="centered")

# SIDEBAR
st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.title("📰 Fake News Detector")

st.sidebar.markdown("---")

st.sidebar.markdown("""
### About
Fake news detector app

### Usage
- Enter news  
- Click button  
- View result  
""")

st.sidebar.markdown("---")
st.sidebar.write("Made by Gurubalan 🚀")

# MAIN PAGE
st.title("📰 Fake News Detector")
st.write("Check whether a news article is real or fake")

st.markdown("""
### Usage
1. Enter news article  
2. Click **Check News**  
3. See prediction + confidence  
""")

# LOAD MODEL
try:
    vectorizer = joblib.load("vectorizer.jb")
    model = joblib.load("lr_model.jb")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

news_input = st.text_area("News Article:")

if st.button("Check News"):
    if news_input.strip():

        with st.spinner("Analyzing news..."):
            transform_input = vectorizer.transform([news_input])
            prediction = model.predict(transform_input)

            proba = model.predict_proba(transform_input)
            real_prob = proba[0][1] * 100
            fake_prob = proba[0][0] * 100

        # 👇 OUTSIDE spinner (important)
        if prediction[0] == 1:
            st.success("The News is Real! ✅")
        else:
            st.error("The News is Fake! ❌")

        st.write(f"Real: {real_prob:.2f}%")
        st.write(f"Fake: {fake_prob:.2f}%")

    else:
        st.warning("Please enter some text")
