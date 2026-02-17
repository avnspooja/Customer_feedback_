import pandas as pd
from textblob import TextBlob
 
def get_sentiment(text):
    polarity =TextBlob(str(text)).sentiment.polarity
    if polarity >0:
        return "positive", polarity
    elif polarity <0:
        return "negative", polarity
    else:
        return "neutral", polarity
 
if __name__ == "__main__":
    df = pd.read_csv("Milestone1_cleaned_feedback.csv")
 
    df[["sentiment", "confidence_score"]] = df["clean_feedback"].apply( lambda x:pd.Series(get_sentiment(x)))
 
    df.to_csv("Milestone2_Sentiment_Results_new.csv", index=False)
 
    print("Milestone 2 completed successfully!")
 
    print(df[["clean_feedback", "sentiment", "confidence_score"]].head())