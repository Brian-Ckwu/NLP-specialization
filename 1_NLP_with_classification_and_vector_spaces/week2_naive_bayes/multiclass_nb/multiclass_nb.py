from typing import Dict, Iterable, List, Set, Tuple
from functools import reduce
import numpy as np
import pandas as pd

from preprocessor import Preprocessor

class NaiveBayes(object):
    def __init__(self, preprocessor: Preprocessor):
        self._preprocessor = preprocessor
        # data
        self._data = None
        self._label2int = dict()
        self._int2label = list()
        # training results
        self._V = set()
        self._priorprob = list()
        self._likelihood = list()
    
    @property
    def data(self) -> Dict[str, np.ndarray]:
        return self._data
    
    @property
    def priorprob(self) -> List[float]:
        return self._priorprob
    
    @property
    def likelihood(self) -> List[Dict[str, float]]:
        return self._likelihood
    
    def map_labels(self, labels: Iterable) -> None:
        for label in labels:
            if label not in self._label2int:
                self._label2int[label] = len(self._label2int)
                self._int2label.append(label)
    
    def convert_labels(self, labels: Iterable) -> np.ndarray:
        convert = np.vectorize(lambda s: self._label2int[s])
        return convert(labels)

    def load(self, train_xs, train_ys, test_xs, test_ys) -> "NaiveBayes":
        # self.__init__(self._preprocessor)
        self.map_labels(train_ys)
        self._data = {
            "train_xs": train_xs,
            "train_ys": self.convert_labels(train_ys),
            "test_xs": test_xs,
            "test_ys": self.convert_labels(test_ys)
        }
        return self
    
    def preprocess_arr(self, arr: Iterable) -> List[List[str]]:
        # preprocess the xs
        l = list()
        for x in arr:
            l.append(self._preprocessor.preprocess(x))
        return l
    
    def build_log_prior(self) -> "NaiveBayes":
        self._priorprob = [0 for _ in range(len(self._int2label))]
        for y in self._data["train_ys"]:
            self._priorprob[y] += 1
        self._priorprob = [np.log(ss / sum(self._priorprob)) for ss in self._priorprob]
        return self
    
    def build_vocab(self, xs: List[List[str]]) -> "NaiveBayes":
        for x in xs:
            for token in x:
                self._V.add(token)
        return self

    def build_log_likelihood(self) -> "NaiveBayes":
        xs = self.preprocess_arr(self._data["train_xs"])
        self.build_vocab(xs)
        word_freqs = [dict.fromkeys(self._V, 0) for _ in range(len(self._int2label))]
        class_freqs = [0 for _ in range(len(self._int2label))]
        for x, y in zip(xs, self._data["train_ys"]):
            for token in x:
                word_freqs[y][token] += 1
                class_freqs[y] += 1
        self._likelihood = [dict.fromkeys(self._V, 0) for _ in range(len(self._int2label))]
        for i in range(len(self._likelihood)):
            for word in self._V:
                self._likelihood[i][word] = np.log((word_freqs[i][word] + 1) / (class_freqs[i] + len(self._V)))
        return self
    
    def train(self):
        return self.build_log_prior().build_log_likelihood()
    
    def predict(self, text: str) -> int:
        n = len(self._int2label)
        probs = [0 for _ in range(n)]
        tokens = self._preprocessor.preprocess(text)
        for i in range(n):
            probs[i] += self._priorprob[i]
            for token in tokens:
                if token in self._V:
                    probs[i] += self._likelihood[i][token]
        return max(enumerate(probs), key=lambda t: t[1])[0]

    def test(self):
        xs = self._data["test_xs"]
        ys = self._data["test_ys"]
        y_hat = list()
        for x, y in zip(xs, ys):
            y_hat.append(self.predict(x))
        y_hat = np.array(y_hat)
        print(f"Accuracy: {np.mean(y_hat == ys)}")
        return self

if __name__ == "__main__":
    import re
    chpattern = r"[\u4e00-\u9fff]+"
    def remove_ch(tokens: List[str]) -> List[str]:
        return list(filter(lambda t: not re.search(pattern=chpattern, string=t), tokens))

    p = Preprocessor()
    p.add_pipe(p.remove_digits).add_pipe(p.remove_not_alpha).add_pipe(remove_ch)

    nb = NaiveBayes(preprocessor=p)

    # load training data
    data = "./data/emr_data_10_icds.tsv"
    with open(data, encoding="utf-8") as f:
        df = pd.read_csv(f, sep="\t", index_col="index")

    train = df.groupby("icd").sample(frac=0.8, random_state=1)
    test = df.drop(train.index)

    nb.load(
        train_xs=train["text"].values,
        train_ys=train["icd"].values,
        test_xs=test["text"].values,
        test_ys=test["icd"].values
    ).train().test()