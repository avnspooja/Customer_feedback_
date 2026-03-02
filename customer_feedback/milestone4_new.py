# ============================
# ReviewSense – Milestone 4 (Fully Fixed Final)
# Interactive Customer Feedback Dashboard
# ============================

import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from wordcloud import WordCloud
import numpy as np
from io import StringIO

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="ReviewSense Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "Milestone2_Sentiment_Results_new.csv")

    if not os.path.exists(file_path):
        st.error(f"❌ File not found: {file_path}")
        st.stop()

    df = pd.read_csv(file_path)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Normalize sentiment
    if "sentiment" in df.columns:
        df["sentiment"] = df["sentiment"].astype(str).str.strip().str.capitalize()

    # Auto-detect date column
    possible_dates = ["date", "Date", "review_date", "timestamp", "created_at"]
    date_col = None
    for col in possible_dates:
        if col in df.columns:
            date_col = col
            break

    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        st.error("❌ No valid date column found in CSV.")
        st.stop()

    return df


@st.cache_data
def load_keywords():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "Milestone3_Keyword_Insights.csv")

    if not os.path.exists(file_path):
        return pd.DataFrame()

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if "=== KEYWORD FREQUENCY ===" in content:
            keyword_part = content.split("=== KEYWORD FREQUENCY ===")[1] \
                                  .split("=== PRODUCT SENTIMENT SUMMARY ===")[0]

            keyword_lines = keyword_part.strip().splitlines()

            if len(keyword_lines) > 1:
                keywords_df = pd.read_csv(StringIO("\n".join(keyword_lines)))
                keywords_df.columns = keywords_df.columns.str.strip()
                return keywords_df

        return pd.DataFrame()

    except:
        return pd.DataFrame()


df = load_data()
keywords_df = load_keywords()

# ----------------------------
# Sidebar Filters
# ----------------------------
st.sidebar.header("🔍 Filters")

sentiment_options = sorted(df["sentiment"].dropna().unique())

sentiment_filter = st.sidebar.multiselect(
    "Select Sentiment",
    options=sentiment_options,
    default=sentiment_options
)

product_filter = st.sidebar.multiselect(
    "Select Product",
    options=sorted(df["product"].dropna().unique()),
    default=sorted(df["product"].dropna().unique())
)

st.sidebar.subheader("📅 Date Range")

if df["date"].notna().any():
    default_start = df["date"].min().date()
    default_end = df["date"].max().date()
else:
    default_start = datetime(2025, 1, 1).date()
    default_end = datetime(2025, 12, 31).date()

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start Date", value=default_start)
end_date = col2.date_input("End Date", value=default_end)

# ----------------------------
# Apply Filters
# ----------------------------
filtered_df = df[
    (df["sentiment"].isin(sentiment_filter)) &
    (df["product"].isin(product_filter)) &
    (df["date"] >= pd.to_datetime(start_date)) &
    (df["date"] <= pd.to_datetime(end_date))
].copy()

# ----------------------------
# Main Dashboard
# ----------------------------
st.markdown('<h1 class="main-header">📊 ReviewSense - Customer Feedback Dashboard</h1>',
            unsafe_allow_html=True)

# ----------------------------
# Key Metrics
# ----------------------------
col1, col2, col3, col4 = st.columns(4)

total_reviews = len(filtered_df)
pos_count = len(filtered_df[filtered_df['sentiment'] == 'Positive'])
neg_count = len(filtered_df[filtered_df['sentiment'] == 'Negative'])
neu_count = len(filtered_df[filtered_df['sentiment'] == 'Neutral'])

def safe_pct(count):
    return (count / total_reviews * 100) if total_reviews > 0 else 0

with col1:
    st.metric("Total Reviews", total_reviews)

with col2:
    st.metric("Positive", f"{safe_pct(pos_count):.1f}%", delta=f"{pos_count} reviews")

with col3:
    st.metric("Negative", f"{safe_pct(neg_count):.1f}%", delta=f"{neg_count} reviews")

with col4:
    st.metric("Neutral", f"{safe_pct(neu_count):.1f}%", delta=f"{neu_count} reviews")

# ----------------------------
# Sentiment Distribution
# ----------------------------
st.subheader("😊 Sentiment Distribution")

if not filtered_df.empty:
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = filtered_df["sentiment"].value_counts()
    counts.plot(kind="bar", ax=ax)
    st.pyplot(fig)
else:
    st.warning("No data matches the selected filters.")

# ----------------------------
# Product Sentiment Table
# ----------------------------
st.subheader("📱 Product-wise Sentiment")

if not filtered_df.empty:
    product_sent = (
        filtered_df.groupby('product')['sentiment']
        .value_counts()
        .unstack(fill_value=0)
    )
    st.dataframe(product_sent, use_container_width=True)

# ----------------------------
# Trend Over Time
# ----------------------------
st.subheader("📈 Sentiment Trends Over Time")

if not filtered_df.empty:
    filtered_df['month'] = filtered_df['date'].dt.to_period('M')
    trend = filtered_df.groupby(['month', 'sentiment']).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    for col in trend.columns:
        ax.plot(trend.index.astype(str), trend[col], marker='o', label=col)

    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

# ----------------------------
# Keywords
# ----------------------------
st.subheader("🔑 Top Keywords")

if not keywords_df.empty:
    st.dataframe(keywords_df.head(15), use_container_width=True)

# ----------------------------
# Confidence Score
# ----------------------------
if "confidence_score" in filtered_df.columns:
    st.subheader("📊 Confidence Score Distribution")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(filtered_df["confidence_score"], bins=25)
    st.pyplot(fig)

# ----------------------------
# Data Preview
# ----------------------------
with st.expander("📋 Preview Data"):
    st.dataframe(filtered_df.head(15), use_container_width=True)

st.success("✅ Dashboard ready!")