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

def plot_confidence_ellipses(data: pd.DataFrame) -> None:
    # Plot the samples using columns 1 and 2 of the matrix
    fig, ax = plt.subplots(figsize = (8, 8))
    colors = ['red', 'green'] # Color base on sentiment
    ax.scatter(data.positive, data.negative, c=[colors[int(k)] for k in data.sentiment], s = 0.1, marker='*')  # Plot a dot for tweet
    # Custom limits for this chart
    plt.xlim(-200,40)  
    plt.ylim(-200,40)

    plt.xlabel("Positive") # x-axis label
    plt.ylabel("Negative") # y-axis label

    data_pos = data[data.sentiment == 1] # Filter only the positive samples
    data_neg = data[data.sentiment == 0] # Filter only the negative samples

    # Print confidence ellipses of 2 std
    confidence_ellipse(data_pos.positive, data_pos.negative, ax, n_std=2, edgecolor='black', label=r'$2\sigma$' )
    confidence_ellipse(data_neg.positive, data_neg.negative, ax, n_std=2, edgecolor='orange')

    # Print confidence ellipses of 3 std
    confidence_ellipse(data_pos.positive, data_pos.negative, ax, n_std=3, edgecolor='black', linestyle=':', label=r'$3\sigma$')
    confidence_ellipse(data_neg.positive, data_neg.negative, ax, n_std=3, edgecolor='orange', linestyle=':')
    ax.legend()

    plt.show()    

if __name__ == "__main__":
    data = pd.read_csv("./bayes_features.csv")
    plot_confidence_ellipses(data) # we can observe whether our model will perform well by the confidence ellipses plot