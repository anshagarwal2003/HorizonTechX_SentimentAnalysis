import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# folders create karna
os.makedirs("dataset", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

data_file = "dataset/customer_reviews.csv"

# agar dataset nahi hai to sample dataset create ho jayega
if not os.path.exists(data_file):
    reviews = {
        "review_id": list(range(1, 61)),
        "customer_review": [
            "The product quality is excellent and I am very happy",
            "Very bad experience, the product stopped working",
            "Average product, not too good and not too bad",
            "I loved the packaging and fast delivery",
            "Waste of money, very poor quality",
            "Good product for this price range",
            "The item is okay but delivery was late",
            "Amazing experience and great value",
            "Not satisfied with the service",
            "Product is decent and works fine",
            "Excellent service and very useful product",
            "Bad quality and not recommended",
            "The product is neutral, nothing special",
            "I am happy with the purchase",
            "Poor packaging and damaged item received",
            "Very useful and easy to use",
            "The quality is not good",
            "Nice product and fast delivery",
            "It is okay for normal use",
            "Worst product I have ever used",
            "Great product and good support",
            "Not worth the price",
            "Satisfied with the performance",
            "The delivery was late but product is fine",
            "Excellent design and premium quality",
            "Bad service from seller",
            "Good quality and value for money",
            "Normal product with average features",
            "I really like this product",
            "Very disappointed with the product",
            "Fast delivery and good packaging",
            "The product works as expected",
            "Poor quality material",
            "Happy with the overall experience",
            "This product is not useful",
            "Average experience with this order",
            "Highly recommended product",
            "Very slow delivery and bad packaging",
            "Good performance and nice look",
            "Not bad, but can be improved",
            "Excellent build quality",
            "I did not like the product",
            "The product is fine for daily use",
            "Very good service",
            "Bad experience with customer support",
            "Amazing quality and smooth performance",
            "Neutral experience, nothing much to say",
            "Poor product and waste of money",
            "Good product but price is high",
            "I am satisfied with this item",
            "Worst packaging ever",
            "Nice and useful product",
            "The item is average",
            "Excellent delivery and good quality",
            "Bad product, not recommended",
            "The product is okay",
            "Very happy with the service",
            "Poor performance after few days",
            "Good value for students",
            "Great experience overall"
        ],
        "rating": [
            5,1,3,5,1,4,3,5,2,3,
            5,1,3,4,1,5,2,4,3,1,
            5,2,4,3,5,1,4,3,5,1,
            4,3,1,5,2,3,5,1,4,3,
            5,2,3,5,1,5,3,1,4,4,
            1,4,3,5,1,3,5,1,4,5
        ]
    }

    df = pd.DataFrame(reviews)
    df.to_csv(data_file, index=False)

# dataset load
df = pd.read_csv(data_file)

print("===== FIRST 5 REVIEWS =====")
print(df.head())

print("\n===== DATASET INFO =====")
print(df.info())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE VALUES =====")
print(df.duplicated().sum())

# duplicate remove
df = df.drop_duplicates()

# missing values handle
df["customer_review"] = df["customer_review"].fillna("No review")
df["rating"] = df["rating"].fillna(df["rating"].mean())

# text clean function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["clean_review"] = df["customer_review"].apply(clean_text)

# simple sentiment logic
positive_words = [
    "good", "great", "excellent", "amazing", "happy", "satisfied",
    "useful", "nice", "loved", "recommended", "value", "premium"
]

negative_words = [
    "bad", "poor", "worst", "waste", "disappointed", "damaged",
    "not satisfied", "not useful", "late", "slow"
]

def get_sentiment(review, rating):
    positive_count = 0
    negative_count = 0

    for word in positive_words:
        if word in review:
            positive_count += 1

    for word in negative_words:
        if word in review:
            negative_count += 1

    if rating >= 4 or positive_count > negative_count:
        return "Positive"
    elif rating <= 2 or negative_count > positive_count:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df.apply(
    lambda row: get_sentiment(row["clean_review"], row["rating"]),
    axis=1
)

df["review_length"] = df["clean_review"].apply(lambda x: len(x.split()))

# save cleaned file
df.to_csv("outputs/cleaned_sentiment_reviews.csv", index=False)

print("\n===== SENTIMENT COUNT =====")
print(df["sentiment"].value_counts())

print("\n===== BASIC STATISTICS =====")
print(df[["rating", "review_length"]].describe())

# chart 1: sentiment count
plt.figure(figsize=(7, 5))
sns.countplot(x="sentiment", data=df)
plt.title("Sentiment Count")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig("outputs/01_sentiment_count.png")
plt.show()

# chart 2: rating distribution
plt.figure(figsize=(7, 5))
sns.countplot(x="rating", data=df)
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig("outputs/02_rating_distribution.png")
plt.show()

# chart 3: rating vs sentiment
plt.figure(figsize=(8, 5))
sns.countplot(x="rating", hue="sentiment", data=df)
plt.title("Rating vs Sentiment")
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig("outputs/03_rating_vs_sentiment.png")
plt.show()

# chart 4: review length distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["review_length"], bins=10, kde=True)
plt.title("Review Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Reviews Count")
plt.tight_layout()
plt.savefig("outputs/04_review_length_distribution.png")
plt.show()

# chart 5: average rating by sentiment
plt.figure(figsize=(7, 5))
sns.barplot(x="sentiment", y="rating", data=df)
plt.title("Average Rating by Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Average Rating")
plt.tight_layout()
plt.savefig("outputs/05_average_rating_by_sentiment.png")
plt.show()

# summary csv
summary = df["sentiment"].value_counts().reset_index()
summary.columns = ["Sentiment", "Count"]
summary.to_csv("outputs/sentiment_summary.csv", index=False)

# report
positive_count = len(df[df["sentiment"] == "Positive"])
negative_count = len(df[df["sentiment"] == "Negative"])
neutral_count = len(df[df["sentiment"] == "Neutral"])

report = f"""
CUSTOMER REVIEW SENTIMENT ANALYSIS REPORT

1. Project Overview
This project analyzes customer reviews and classifies them into Positive, Negative, and Neutral sentiments.

2. Dataset Overview
Total Reviews: {len(df)}
Total Columns: {df.shape[1]}

3. Data Cleaning
- Missing values were checked and handled.
- Duplicate records were removed.
- Review text was converted into lowercase.
- Special characters were removed from reviews.

4. Sentiment Summary
Positive Reviews: {positive_count}
Negative Reviews: {negative_count}
Neutral Reviews: {neutral_count}

5. Rating Statistics
Minimum Rating: {df['rating'].min()}
Maximum Rating: {df['rating'].max()}
Average Rating: {df['rating'].mean():.2f}

6. Key Insights
- Positive reviews show customer satisfaction.
- Negative reviews highlight issues related to quality, delivery, or service.
- Neutral reviews show average customer experience.
- Rating and sentiment are strongly connected.
- Sentiment analysis helps understand customer opinion in a simple way.

7. Conclusion
This project performs text cleaning, sentiment classification, visualization, and report generation.
It helps understand customer feedback and supports better decision-making.
"""

with open("outputs/sentiment_analysis_report.txt", "w") as file:
    file.write(report)

print("\nSentiment Analysis Project completed successfully!")
print("Cleaned dataset saved in outputs folder.")
print("Charts generated successfully.")
print("Report generated successfully.")