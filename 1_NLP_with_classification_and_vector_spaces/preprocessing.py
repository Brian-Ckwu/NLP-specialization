import nltk
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import random

def load_tweets() -> dict:
    tweets = {
        "pos": twitter_samples.strings("positive_tweets.json"),
        "neg": twitter_samples.strings("negative_tweets.json")
    }
    return tweets

def display_tweets_info(tweets: dict) -> None:
    for pon in tweets:
        print(f"Number of {pon}{'itive' if pon == 'pos' else 'ative'} tweets: {len(tweets[pon])}")
    print(f"\nThe type of tweets: {type(tweets[pon])}")
    print(f"The type of a tweet entry: {type(tweets[pon][0])}")

def plot_tweets_ratio(tweets: dict) -> None:
    plt.figure(figsize=(5, 5))
    plt.pie(
        x=[len(tweets[pon]) for pon in tweets],
        labels=["Positive", "Negative"],
        autopct="%1.1f%%",
        startangle=90
    )
    plt.axis("equal")
    plt.show()

if __name__ == "__main__":
    tweets = load_tweets()
    display_tweets_info(tweets)
    plot_tweets_ratio(tweets)