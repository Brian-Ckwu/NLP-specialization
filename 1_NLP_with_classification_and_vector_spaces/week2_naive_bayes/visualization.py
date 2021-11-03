import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import confidence_ellipse

def plot_tweets(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["red", "green"] # red: negative tweets; green: positive tweets
    # scatter plot
    ax.scatter(data["positive"], data["negative"], c=[colors[int(k)] for k in data["sentiment"]], s=0.1)
    # axis limits
    ax.set_xlim(-250, 0)
    ax.set_ylim(-250, 0)
    # axis labels
    ax.set_xlabel("Positive")
    ax.set_ylabel("Negative")
    # show the plot
    plt.show()

if __name__ == "__main__":
    data = pd.read_csv("./bayes_features.csv")
    plot_tweets(data)