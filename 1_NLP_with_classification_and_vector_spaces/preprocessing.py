import nltk
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import random

class Tweets(object):
    def __init__(self):
        # load tweets
        self._tweets = {
            "pos": twitter_samples.strings("positive_tweets.json"),
            "neg": twitter_samples.strings("negative_tweets.json")
        }
    
    def display_info(self):
        for pon in self._tweets:
            print(f"Number of {pon}{'itive' if pon == 'pos' else 'ative'} tweets: {len(self._tweets[pon])}")
        print(f"\nThe type of tweets: {type(self._tweets[pon])}")
        print(f"The type of a tweet entry: {type(self._tweets[pon][0])}")
    
    def plot_pos_neg_ratio(self):
        plt.figure(figsize=(5, 5))
        plt.pie(
            x=[len(self._tweets[pon]) for pon in self._tweets],
            labels=["Positive", "Negative"],
            autopct="%1.1f%%",
            startangle=90
        )
        plt.axis("equal")
        plt.show()        

    # Before anything else, we can print a couple of tweets from the dataset to see how they look.
    # Understanding the data is responsible for 80% of the success or failure in data science projects.
    # We can use this time to observe aspects we'd like to consider when preprocessing our data.
    def display_random_tweets(self):
        green, red = "\033[92m", "\033[91m"
        for pon in self._tweets:
            print(f"{green if pon == 'pos' else red}{self._tweets[pon][random.randrange(0, len(self._tweets[pon]))]}")

    def get_tweet(self, pon, index):
        return self._tweets[pon][index]

"""
    Preprocessing steps:
        1. Tokenizing
        2. Lowercasing
        3. Removing stop words and punctuation
        4. Stemming (lemmatizing?)
        5. Othe customized preprocessing steps
"""

EXAMPLE_TWEET_INDEX = 2277

if __name__ == "__main__":
    tweets = Tweets()
    example = tweets.get_tweet(pon="pos", index=EXAMPLE_TWEET_INDEX)
    print(example)