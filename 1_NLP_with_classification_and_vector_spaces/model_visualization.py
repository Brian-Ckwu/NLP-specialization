import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

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
    for i, x in enumerate(train_xs):
        tokens = p.process_tweet(x)
        tweet = extract_features(tokens, word_freqs)
        tweets.append(tweet + [1.0 if i < 4000 else 0.0])
    # DataFrame
    df = pd.DataFrame(tweets, columns=["bias", "pos", "neg", "sentiment"])
    print(df)