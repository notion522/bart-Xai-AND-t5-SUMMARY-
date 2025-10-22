import streamlit as st
from summarizer_core import summarize_text, compute_bart_attributions

st.set_page_config(page_title="Text Summarizer + Explainability", layout="wide")
st.title("Text Summarizer + Explainability")

model_choice = st.selectbox("Choose model", ["BART", "T5"])
user_input = st.text_area("Enter text to summarize:", height=200)

if st.button("Summarize"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        model_name = "facebook/bart-large-cnn" if model_choice == "BART" else "t5-small"

        with st.spinner("Summarizing..."):
            summary = summarize_text(user_input, model_name)

        st.subheader("📋 Summary:")
        st.write(summary)

        if model_choice == "BART":
            with st.spinner("Computing token-level attributions..."):
                attributions = compute_bart_attributions(user_input)
            st.subheader("🔍 Token-level Attributions :")
            st.markdown("| Token -> Attribution |\n")
            for token, score in attributions:
                st.markdown(f"| `{token:}` -> {score:.6f} |")
