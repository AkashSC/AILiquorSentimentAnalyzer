# app.py
import os
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# Config
# ===============================
st.set_page_config(page_title="🍷 Liquor Market Sentiment AI", layout="wide")

# Custom CSS for smaller font + wrap text
st.markdown("""
    <style>
    body, div, p, input, textarea, span, label {
        font-size: 13px !important;
    }
    .stDataFrame div[data-testid="stDataFrame"] {
        font-size: 12px !important;
        white-space: normal !important;
    }
    </style>
""", unsafe_allow_html=True)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"

# ===============================
# Helper: Call Groq API
# ===============================
def analyze_sentiment(feedback: str) -> dict:
    if not GROQ_API_KEY:
        return {"error": "Missing GROQ_API_KEY"}

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a sentiment analysis assistant for liquor market feedback."},
            {"role": "user", "content": f"""
            Analyse the following feedback and return a JSON with fields:
            overall_sentiment (positive/neutral/negative),
            sentiment_score (-1 to 1),
            top_aspects (list of main points),
            short_actionable_insight (1–2 lines).
            
            Feedback: '''{feedback}'''
            """}
        ],
        "temperature": 0.0
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return {"raw": content}
    except Exception as e:
        return {"error": str(e)}

# ===============================
# Data store (in session)
# ===============================
if "feedback_data" not in st.session_state:
    st.session_state.feedback_data = []

# ===============================
# Tabs
# ===============================
tab1, tab2 = st.tabs(["📝 Feedback Input", "📊 Sentiment Dashboard"])

# -------------------------------
# TAB 1: Input
# -------------------------------
with tab1:
    st.header("Enter Liquor Brand Feedback")

    col1, col2 = st.columns(2)
    with col1:
        brand = st.text_input("Brand Name", placeholder="e.g., Jack Daniels")
    with col2:
        flavor = st.text_input("Flavor / Variant", placeholder="e.g., Honey, Classic")

    feedback_text = st.text_area("Customer Feedback", placeholder="Enter feedback here...", height=120)

    if st.button("🔍 Analyze Sentiment"):
        if brand and feedback_text:
            result = analyze_sentiment(feedback_text)
            st.session_state.feedback_data.append({
                "brand": brand,
                "flavor": flavor,
                "feedback": feedback_text,
                "result": result
            })
            st.success("Feedback submitted and analyzed!")
        else:
            st.warning("Please enter both Brand and Feedback.")

# -------------------------------
# TAB 2: Dashboard
# -------------------------------
with tab2:
    st.header("📊 Brand Sentiment Dashboard")

    if not st.session_state.feedback_data:
        st.info("No feedback yet. Add some in the first tab.")
    else:
        # Keep table
        df = pd.DataFrame([
            {
                "Brand": item["brand"],
                "Flavor": item["flavor"],
                "Feedback & Analysis": f"Feedback: {item['feedback']}\n\nAnalysis: {item['result'].get('raw', str(item['result']))}"
            }
            for item in st.session_state.feedback_data
        ])

        st.subheader("📋 Recent Feedback & Analysis")
        st.dataframe(df, use_container_width=True, height=250)

        # Compact sentiment chart
        st.subheader("📈 Brand Sentiment Trend")

        sentiment_summary = []
        for item in st.session_state.feedback_data:
            analysis = str(item["result"].get("raw", "")).lower()
            if "positive" in analysis:
                sentiment_summary.append((item["brand"], "Positive"))
            elif "negative" in analysis:
                sentiment_summary.append((item["brand"], "Negative"))
            else:
                sentiment_summary.append((item["brand"], "Neutral"))

        trend_df = pd.DataFrame(sentiment_summary, columns=["Brand", "Sentiment"])
        counts = trend_df.groupby(["Brand", "Sentiment"]).size().unstack(fill_value=0)

       # Place chart in a medium column
        col_chart, _ = st.columns([2, 1])  # chart takes medium space
        with col_chart:
            fig, ax = plt.subplots(figsize=(4.5, 3), dpi=120)  # medium box

            colors = {"Positive": "green", "Negative": "red", "Neutral": "gray"}

            # Convert counts to percentages
            percent_df = counts.div(counts.sum(axis=1), axis=0) * 100

            percent_df.plot(
                kind="bar",
                stacked=True,
                ax=ax,
                width=0.35,   # thinner bars
                color=[colors.get(sent, "blue") for sent in percent_df.columns],
                legend=True
            )

            # Show sentiment % labels OUTSIDE the bar box
            for container in ax.containers:
                for bar in container:
                    value = bar.get_height()
                    if value > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_y() + bar.get_height() + 1,   # position ABOVE bar
                            f"{value:.0f}%",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            color="black"
                        )

            ax.set_ylabel("Share (%)", fontsize=8)
            ax.set_title("Sentiment by Brand", fontsize=9)
            plt.xticks(rotation=30, ha="right", fontsize=7)
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)

