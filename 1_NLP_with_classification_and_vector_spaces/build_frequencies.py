import nltk
import matplotlib.pyplot as plt
import numpy as np

from preprocessing import Tweets, TweetPreprocessor
from typing import List, Tuple, Dict

SELECTED_WORDS = ['happi', 'merri', 'nice', 'good', 'bad', 'sad', 'mad', 'best', 'pretti',
        '❤', ':)', ':(', '😒', '😬', '😄', '😍', '♛',
        'song', 'idea', 'power', 'play', 'magnific']

def build_freqs(tweets: List[str], ys: List[float], p: TweetPreprocessor) -> Dict[Tuple[str, float], int]:
    freqs = dict()
    for tweet, label in zip(tweets, ys):
        tokens = p.process_tweet(tweet)
        for token in tokens:
            t = (token, label)
            freqs[t] = freqs.get(t, 0) + 1 # a more concise coding than the following lines
            # if t in freqs:
            #     freqs[t] += 1
            # else:
            #     freqs[t] = 1
    return freqs

def count_words_freqs(words: List[str], freqs: Dict[Tuple[str, float], int]) -> List[list]:
    words_freqs = list()
    for word in words:
        pos = freqs.get((word, 1), 0)
        neg = freqs.get((word, 0), 0)
        words_freqs.append([word, pos, neg])
    return words_freqs

if __name__ == "__main__":
    t = Tweets()
    p = TweetPreprocessor()
    # build frequencies
    tweets = t.tweets["pos"] + t.tweets["neg"]
    ys = np.append(np.ones(len(t.tweets["pos"])), np.zeros(len(t.tweets["neg"])))
    freqs = build_freqs(tweets, ys, p)
    # visualize the frequencies of selected words
    words_freqs = count_words_freqs(SELECTED_WORDS, freqs)
    print(words_freqs)