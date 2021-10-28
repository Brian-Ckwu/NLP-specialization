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

# Before anything else, we can print a couple of tweets from the dataset to see how they look.
# Understanding the data is responsible for 80% of the success or failure in data science projects.
# We can use this time to observe aspects we'd like to consider when preprocessing our data.
def display_random_tweets(tweets: dict) -> None:
    green, red = "\033[92m", "\033[91m"
    for pon in tweets:
        print(f"{green if pon == 'pos' else red}{tweets[pon][random.randrange(0, len(tweets[pon]))]}")

if __name__ == "__main__":
    tweets = load_tweets()
    display_tweets_info(tweets)
    # plot_tweets_ratio(tweets)
    display_random_tweets(tweets)