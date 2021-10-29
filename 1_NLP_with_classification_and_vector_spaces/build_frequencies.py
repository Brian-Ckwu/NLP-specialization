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

def plot_words_freqs(words_freqs: List[list]) -> None:
    fig, ax = plt.subplots(figsize=(8, 8)) # ax: a single subplot / axs: multiple subplots
    # words, pos_freqs, neg_freqs
    words = [l[0] for l in words_freqs]
    xs = np.log([l[1] + 1 for l in words_freqs]) # logarithmic to avoid large margins # + 1 to avoid log(0)
    ys = np.log([l[2] + 1 for l in words_freqs])
    # scatter plot
    ax.scatter(xs, ys)
    ax.set_xlabel("Log Positive Count")
    ax.set_ylabel("Log Negative Count")
    for i in range(len(words_freqs)):
        ax.annotate(words[i], (xs[i], ys[i]), fontsize=12) # annotate the coordinates with tweet words
    # show the scatter plot
    ax.plot([0, 9], [0, 9], color="red")
    plt.show()

if __name__ == "__main__":
    t = Tweets()
    p = TweetPreprocessor()
    # build frequencies
    tweets = t.tweets["pos"] + t.tweets["neg"]
    ys = np.append(np.ones(len(t.tweets["pos"])), np.zeros(len(t.tweets["neg"])))
    freqs = build_freqs(tweets, ys, p)
    # visualize the frequencies of selected words
    words_freqs = count_words_freqs(SELECTED_WORDS, freqs)
    plot_words_freqs(words_freqs)