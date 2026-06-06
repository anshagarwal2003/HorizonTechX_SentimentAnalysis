import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# page setup
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="💬",
    layout="wide"
)

# css for dashboard design
st.markdown("""
<style>
.stApp {
    background: #0f172a;
    color: #e5e7eb;
}

.title-box {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    padding: 28px;
    border-radius: 18px;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.35);
}

.title-box h1 {
    color: white;
    font-size: 38px;
    margin-bottom: 8px;
}

.title-box p {
    color: #e0e7ff;
    font-size: 16px;
    margin: 0px;
}

.card {
    background: #1e293b;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    border: 1px solid #334155;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.25);
}

.card h3 {
    color: #93c5fd;
    font-size: 16px;
    margin-bottom: 8px;
}

.card h2 {
    color: white;
    font-size: 32px;
    margin: 0px;
}

.section {
    background: #1e293b;
    padding: 13px 16px;
    border-radius: 14px;
    margin-top: 24px;
    margin-bottom: 15px;
    border-left: 6px solid #60a5fa;
}

.section h3 {
    color: white;
    margin: 0px;
}

.info-box {
    background: #111827;
    padding: 18px;
    border-radius: 14px;
    border-left: 6px solid #22c55e;
    color: #e5e7eb;
    font-size: 16px;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.25);
}

.insight-box {
    background: #111827;
    padding: 18px;
    border-radius: 14px;
    border-left: 6px solid #facc15;
    color: #e5e7eb;
}

.insight-box li {
    margin-bottom: 8px;
}

.chart-title {
    color: #e5e7eb;
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 8px;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

[data-testid="stSidebar"] * {
    color: #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

# dataset load
df = pd.read_csv("outputs/cleaned_sentiment_reviews.csv")

# main heading
st.markdown("""
<div class="title-box">
    <h1>💬 Customer Review Sentiment Analysis</h1>
    <p>Analyze customer feedback using ratings, review text and sentiment categories</p>
</div>
""", unsafe_allow_html=True)

# sidebar filters
st.sidebar.header("🔍 Filter Reviews")
st.sidebar.write("Use filters to explore customer feedback.")

sentiment_filter = st.sidebar.multiselect(
    "Select Sentiment",
    df["sentiment"].unique(),
    default=df["sentiment"].unique()
)

rating_filter = st.sidebar.multiselect(
    "Select Rating",
    sorted(df["rating"].unique()),
    default=sorted(df["rating"].unique())
)

# filtered data
filtered_df = df[
    (df["sentiment"].isin(sentiment_filter)) &
    (df["rating"].isin(rating_filter))
]

if filtered_df.empty:
    st.warning("No reviews found for selected filters.")
    st.stop()

# values for cards
total_reviews = len(filtered_df)
positive_reviews = len(filtered_df[filtered_df["sentiment"] == "Positive"])
negative_reviews = len(filtered_df[filtered_df["sentiment"] == "Negative"])
neutral_reviews = len(filtered_df[filtered_df["sentiment"] == "Neutral"])
avg_rating = filtered_df["rating"].mean()

positive_percent = (positive_reviews / total_reviews) * 100
negative_percent = (negative_reviews / total_reviews) * 100

# cards
st.markdown('<div class="section"><h3>📌 Review Summary</h3></div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="card">
        <h3>Total Reviews</h3>
        <h2>{total_reviews}</h2>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="card">
        <h3>Positive</h3>
        <h2>{positive_reviews}</h2>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="card">
        <h3>Negative</h3>
        <h2>{negative_reviews}</h2>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="card">
        <h3>Neutral</h3>
        <h2>{neutral_reviews}</h2>
    </div>
    """, unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="card">
        <h3>Avg Rating</h3>
        <h2>{avg_rating:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

# quick observation
st.markdown('<div class="section"><h3>📍 Quick Observation</h3></div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="info-box">
    From the selected reviews, <b>{positive_percent:.1f}%</b> reviews are positive and 
    <b>{negative_percent:.1f}%</b> reviews are negative. This gives a quick idea of customer satisfaction.
</div>
""", unsafe_allow_html=True)

# charts
st.markdown('<div class="section"><h3>📊 Sentiment Visualizations</h3></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="chart-title">Sentiment Count</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="sentiment", data=filtered_df, ax=ax)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Reviews")
    ax.set_facecolor("#f8fafc")
    st.pyplot(fig)

with col2:
    st.markdown('<div class="chart-title">Rating Distribution</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="rating", data=filtered_df, ax=ax)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of Reviews")
    ax.set_facecolor("#f8fafc")
    st.pyplot(fig)

col3, col4 = st.columns(2)

with col3:
    st.markdown('<div class="chart-title">Rating vs Sentiment</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="rating", hue="sentiment", data=filtered_df, ax=ax)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Reviews Count")
    ax.set_facecolor("#f8fafc")
    st.pyplot(fig)

with col4:
    st.markdown('<div class="chart-title">Review Length Distribution</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(filtered_df["review_length"], bins=10, kde=True, ax=ax)
    ax.set_xlabel("Review Length")
    ax.set_ylabel("Reviews Count")
    ax.set_facecolor("#f8fafc")
    st.pyplot(fig)

# average rating chart
st.markdown('<div class="section"><h3>⭐ Average Rating by Sentiment</h3></div>', unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x="sentiment", y="rating", data=filtered_df, ax=ax)
ax.set_xlabel("Sentiment")
ax.set_ylabel("Average Rating")
ax.set_facecolor("#f8fafc")
st.pyplot(fig)

# table
st.markdown('<div class="section"><h3>📄 Filtered Customer Reviews</h3></div>', unsafe_allow_html=True)

show_cols = ["review_id", "customer_review", "rating", "sentiment", "review_length"]
st.dataframe(filtered_df[show_cols], use_container_width=True)

# download button
csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇ Download Filtered Reviews",
    data=csv_data,
    file_name="filtered_reviews.csv",
    mime="text/csv"
)

# insights
st.markdown('<div class="section"><h3>💡 Key Insights</h3></div>', unsafe_allow_html=True)

st.markdown("""
<div class="insight-box">
<ul>
    <li>Positive reviews show customer satisfaction with product quality, delivery or service.</li>
    <li>Negative reviews help identify problems like poor quality, late delivery or bad support.</li>
    <li>Neutral reviews show average customer experience.</li>
    <li>Rating distribution helps understand overall customer opinion.</li>
    <li>This dashboard can help businesses improve product quality and customer service.</li>
</ul>
</div>
""", unsafe_allow_html=True)