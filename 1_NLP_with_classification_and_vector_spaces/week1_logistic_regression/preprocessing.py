from typing import List, Dict
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
    
    @property
    def tweets(self) -> Dict[str, list]:
        return self._tweets
    
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
        * Other customized preprocessing steps appropriate for the task
"""

import re
import string

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class TweetPreprocessor(object):
    def __init__(self):
        self._tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
        self._stopwords = set(stopwords.words("english"))
        self._punctuations = set(string.punctuation)
        self._stemmer = PorterStemmer()

    # Remove hyperlinks, Twitter marks and styles
    def customized_preprocess(self, tweet: str) -> str:
        tweet2 = re.sub(r'^RT[\s]+', '', tweet) # remove old style retweet text "RT"
        tweet2 = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet2) # remove hyperlinks
        tweet2 = re.sub(r'#', '', tweet2) # remove hashtags: only removing the hash # sign from the word
        return tweet2

    def tokenize(self, tweet: str) -> List[str]:
        return self._tokenizer.tokenize(tweet)

    def remove_sw_punc(self, tokens: List[str]) -> List[str]:
        clean_tokens = list()
        for token in tokens:
            if (token not in self._stopwords) and (token not in self._punctuations):
                clean_tokens.append(token)
        return clean_tokens
    
    def stem(self, word: str) -> str:
        return self._stemmer.stem(word)
    
    def process_tweet(self, tweet: str) -> List[str]:
        tweet = self.customized_preprocess(tweet)
        tokens = self.tokenize(tweet)
        clean_tokens = self.remove_sw_punc(tokens)
        stemmed_tokens = list()
        for clean_token in clean_tokens:
            stemmed_tokens.append(self.stem(clean_token))
        return stemmed_tokens

EXAMPLE_TWEET_INDEX = 2277

if __name__ == "__main__":
    tweets = Tweets()
    preprocessor = TweetPreprocessor()
    example = tweets.get_tweet(pon="pos", index=EXAMPLE_TWEET_INDEX)
    print(example, end="\n\n")
    pp_tokens = preprocessor.process_tweet(example)
    print(pp_tokens)