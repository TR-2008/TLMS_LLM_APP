import io
import os
import pandas as pd
import streamlit as st
from groq import Groq

st.set_page_config(page_title="Retail AI Analyst", page_icon="📊", layout="wide")
st.title("📊 Retail AI Analyst")
st.caption("Upload a monthly sales CSV and ask an LLM to explain evidence-based trends.")

with st.sidebar:
    st.header("Configuration")
    model = st.text_input("Groq model ID", value="llama-3.3-70b-versatile",
                          help="Use a model ID currently available to your Groq account.")
    st.caption("The API key is read from Streamlit Secrets or the GROQ_API_KEY environment variable.")

api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
uploaded = st.file_uploader("Upload sales CSV", type=["csv"])
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        st.subheader("Dataset preview")
        st.dataframe(df.head(20), use_container_width=True)
        st.write(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
        st.write("Missing values")
        st.dataframe(df.isna().sum().rename("Missing values"))
        st.subheader("Numeric summary")
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            metric = st.selectbox("Choose a numeric column to summarize", numeric_cols)
            st.metric(f"Total {metric}", f"{df[metric].sum():,.2f}")
            st.metric(f"Average {metric}", f"{df[metric].mean():,.2f}")
            if len(numeric_cols) >= 1:
                st.bar_chart(df[numeric_cols].mean().to_frame("Mean"))

        question = st.text_area(
            "Ask about this dataset",
            value="Summarize the key trends, unusual values, and practical actions. Distinguish facts from possible explanations."
        )
        if st.button("Generate AI analysis", type="primary"):
            if not api_key:
                st.error("Missing Groq API key. Add GROQ_API_KEY in Streamlit Cloud Secrets or your local environment.")
            elif not question.strip():
                st.warning("Enter a question first.")
            else:
                # Limit rows sent to the model and avoid sending obvious personal data.
                safe_df = df.head(100)
                csv_context = safe_df.to_csv(index=False)
                prompt = (
                    "Analyze this retail dataset carefully. Use only the supplied data. "
                    "Do not invent causes or facts. Separate observations from hypotheses, "
                    "mention that only the first 100 rows are provided if applicable, and "
                    "give three measurable recommendations.\n\n"
                    f"Question: {question}\n\nCSV data (first 100 rows):\n{csv_context}"
                )
                try:
                    client = Groq(api_key=api_key)
                    with st.spinner("Analyzing with Groq..."):
                        response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": "You are a careful retail data analyst. Never fabricate statistics."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.2,
                            max_completion_tokens=1200
                        )
                    st.subheader("AI-generated analysis")
                    st.markdown(response.choices[0].message.content or "The model returned an empty response.")
                except Exception as exc:
                    st.error(f"Groq request failed: {exc}")
    except Exception as exc:
        st.error(f"Could not read this CSV: {exc}")
else:
    st.info("Upload a CSV file to begin. For best results include columns such as Month, Region, Sales, and Orders.")
    st.markdown("**Privacy:** Do not upload customer names, phone numbers, addresses, payment data, or other sensitive personal information.")
