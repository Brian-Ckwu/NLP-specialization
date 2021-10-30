import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from tqdm import tqdm

from preprocessing import Tweets, TweetPreprocessor
from build_frequencies import build_freqs

def extract_features(tweet_tokens: List[str], word_freqs: Dict[Tuple[str, float], int]) -> List[float]:
    bias = 1.0
    pos = 0.0
    neg = 0.0
    for token in tweet_tokens:
        pos += word_freqs.get((token, 1), 0)
        neg += word_freqs.get((token, 0), 0)
    return [bias, pos, neg]

def neg(theta, pos):
    return (-theta[0] - pos * theta[1]) / theta[2]

def direction(theta, pos):
    return pos * theta[2] / theta[1]

if __name__ == "__main__":
    t = Tweets()
    p = TweetPreprocessor()
    # 8000 tweets for training
    train_xs = t.tweets["pos"][:4000] + t.tweets["neg"][:4000]
    train_ys = np.append(np.ones(4000), np.zeros(4000))
    # build words frequencies hashmap for feature extraction
    word_freqs = build_freqs(train_xs, train_ys, p)
    # extract features
    tweets = list()
    for i, x in tqdm(enumerate(train_xs)):
        tokens = p.process_tweet(x)
        tweet = extract_features(tokens, word_freqs)
        tweets.append(tweet + [1.0 if i < 4000 else 0.0])
    # DataFrame
    df = pd.DataFrame(tweets, columns=["bias", "pos", "neg", "sentiment"])
    X = df[["bias", "pos", "neg"]].values
    Y = df["sentiment"].values
    thetas = [7e-08, 0.0005239, -0.00055517] # parameters of a pretrained logistc regression model
    # Visualization
    # data points
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["red", "green"]
    ax.scatter(X[:, 1], X[:, 2], c=[colors[int(y)] for y in Y], s=0.1)
    ax.set_xlabel("Positive")
    ax.set_ylabel("Negative")
    # logistic model (drawn as a line)
    maxpos = np.max(X[:,1])
    offset = 5000 # The pos value for the direction vectors origin
    # Plot a gray line that divides the 2 areas.
    ax.plot([0, maxpos], [neg(thetas, 0),   neg(thetas, maxpos)], color = 'gray') 
    # Plot a green line pointing to the positive direction
    ax.arrow(offset, neg(thetas, offset), offset, direction(thetas, offset), head_width=500, head_length=500, fc='g', ec='g')
    # Plot a red line pointing to the negative direction
    ax.arrow(offset, neg(thetas, offset), -offset, -direction(thetas, offset), head_width=500, head_length=500, fc='r', ec='r')
    plt.show()