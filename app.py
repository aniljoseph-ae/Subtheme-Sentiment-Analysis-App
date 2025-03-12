'''

import streamlit as st
from utils import load_model_and_tokenizer, analyze_review_detailed
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("aniljoseph/subtheme_sentiment_BERT_finetuned")
model = BertForSequenceClassification.from_pretrained("aniljoseph/subtheme_sentiment_BERT_finetuned")

if tokenizer is None or model is None:
    st.error("Failed to load the tokenizer or model. Please ensure that all dependencies are installed and the model identifier is correct.")
else:
    # --------------- Streamlit App UI ----------------
    st.title("Subtheme Sentiment Analysis")
    st.write("Analyze subthemes and their sentiment in user reviews.")

    # Input field
    user_input = st.text_area("✍️ Please enter your review:")

    if st.button("🔍 Analyze"):
        if user_input:
            result = analyze_review_detailed(user_input, tokenizer, model)
            st.subheader("📌 Results: Subtheme Sentiment Analysis")

            for subtheme, details in result.items():
                st.markdown(f"**🔹 Subtheme:** {subtheme} ({details['subtheme_category']})")
                st.markdown(f"**➤ Sentence:** {details['sentence']}")
                st.markdown(f"**➤ Sentiment:** {details['sentiment']}")
                st.write("---")

            # Debugging output in Streamlit
            st.write("### Debugging Information")
            st.write("Extracted Subthemes:", result)
        else:
            st.warning("⚠️ Please enter a review for analysis.")
'''

import streamlit as st
from util import analyze_review_detailed
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load fine-tuned BERT model
model_name = "aniljoseph/subtheme_sentiment_BERT_finetuned"  # Replace with your actual model name on Hugging Face
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    st.write("✅ Model Loaded Successfully from Hugging Face!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

st.title("Subtheme Sentiment Analysis")
st.write("Analyze subthemes and their sentiment in user reviews.")

user_input = st.text_area("✍️ Please enter your review:")

if st.button("🔍 Analyze"):
    if user_input:
        # result = analyze_review_detailed(user_input, tokenizer, model)  # ✅ Now passing tokenizer and model correctly
        result = analyze_review_detailed(user_input, tokenizer, model)  # ✅ Now correctly passing tokenizer and model

        st.subheader("📌 Results: Subtheme Sentiment Analysis")
        for subtheme, details in result.items():
            st.markdown(f"**🔹 {subtheme} ({details['subtheme_category']})**")
            st.markdown(f"**➤ Sentence:** {details['sentence']}")
            st.markdown(f"**➤ Sentiment:** {details['sentiment']}")
            st.write("---")
        st.write("### Debugging Information")
        st.write("Extracted Subthemes:", result)
    else:
        st.warning("⚠️ Please enter a review for analysis.")
