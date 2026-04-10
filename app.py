import streamlit as st
import joblib
from streamlit_lottie import st_lottie
import requests

def load_lottie(url):
    r = requests.get(url)
    return r.json()

# Success animation (Real news)
lottie_success = load_lottie("https://assets10.lottiefiles.com/packages/lf20_jbrw3hcz.json")

# Error animation (Fake news)
lottie_error = load_lottie("https://assets10.lottiefiles.com/packages/lf20_qp1q7mct.json")

# 🔥 SIDEBAR DESIGN
st.sidebar.image("logo.png", width=200)

st.sidebar.title("Fake News Detector")

st.sidebar.markdown("""
### About
This app uses Machine Learning to classify news as:

- ✅ Real News  
- ❌ Fake News  

### Model
- Logistic Regression  
- TF-IDF Vectorizer  

### Usage
1. Enter news article  
2. Click **Check News**  
3. See prediction + confidence  
""")

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

        # 🔥 ANIMATION + RESULT
        if prediction[0] == 1:
            st_lottie(lottie_success, height=150)
            st.success("The News is Real! ✅")
        else:
            st_lottie(lottie_error, height=150)
            st.error("The News is Fake! ❌")

        st.write(f"Real: {real_prob:.2f}%")
        st.write(f"Fake: {fake_prob:.2f}%")

    else:
        st.warning("Please enter some text")
