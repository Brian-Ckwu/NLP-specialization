import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict

from preprocessing import Tweets, TweetPreprocessor
from build_frequencies import build_freqs
from logistic_regression import sigmoid, gradient_descent

def extract_features(tweet: str, freqs: Dict[Tuple[str, float], int], p: TweetPreprocessor) -> np.ndarray:
    tokens = p.process_tweet(tweet)
    x = np.zeros((1, 3))
    x[0, 0] = 1.0
    for token in tokens:
        x[0, 1] += freqs.get((token, 1), 0)
        x[0, 2] += freqs.get((token, 0), 0)
    return x

def predict_tweet(tweet: str, freqs: Dict[Tuple[str, float], int], p: TweetPreprocessor, theta: np.ndarray) -> float:
    x = extract_features(tweet, freqs, p)
    z = (x @ theta).item()
    h = sigmoid(z)
    return h

def evaluate_accuracy(xs: List[str], ys: np.ndarray, freqs: Dict[Tuple[str, float], int], p: TweetPreprocessor, theta: np.ndarray) -> float:
    preds = list()
    for x in tqdm(xs):
        h = predict_tweet(x, freqs, p, theta)
        if h >= 0.50:
            preds.append(1)
        else:
            preds.append(0)
    y_hat = np.array(preds)
    return np.mean(y_hat == ys)

TRAINING_HALF = 4000
TESTING_HALF = 1000
LEARNING_RATE = 1e-9
ITERATIONS = 1000

if __name__ == "__main__":
    t = Tweets()
    p = TweetPreprocessor()

    train_xs, train_ys = t.tweets["pos"][:TRAINING_HALF] + t.tweets["neg"][:TRAINING_HALF], np.append(np.ones((TRAINING_HALF, 1)), np.zeros((TRAINING_HALF, 1)), axis=0)
    test_xs, test_ys = t.tweets["pos"][TRAINING_HALF:] + t.tweets["neg"][TRAINING_HALF:], np.append(np.ones(TESTING_HALF), np.zeros(TESTING_HALF))
    freqs = build_freqs(train_xs, train_ys.ravel(), p)
    # feature matrix
    m = TRAINING_HALF * 2
    X = np.zeros((m, 3))
    Y = train_ys
    print("Extracting features...")
    for i, x in tqdm(enumerate(train_xs)):
        X[i, :] = extract_features(x, freqs, p)
    # start training
    J, theta = gradient_descent(X, Y, theta=np.zeros((3, 1)), alpha=LEARNING_RATE, num_iters=ITERATIONS)
    print(J, theta.ravel())

    # load model
    # model = "./models/alpha1e8_iter1000.txt"
    # with open(model) as f:
    #     f.readline()
    #     theta = np.array(list(map(float, f.readline().split()))).reshape((3, 1))
    
    acc = evaluate_accuracy(test_xs, test_ys, freqs, p, theta)
    print(acc)