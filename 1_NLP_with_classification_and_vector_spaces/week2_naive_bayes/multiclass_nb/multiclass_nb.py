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
        self._priorprob = list()
        self._likelihood = dict()
    
    @property
    def data(self) -> Dict[str, np.ndarray]:
        return self._data
    
    @property
    def priorprob(self) -> List[float]:
        return self._priorprob
    
    @property
    def likelihood(self) -> Dict[Tuple[str, int], float]:
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
    
    def build_log_likelihood(self) -> "NaiveBayes":
        xs = self.preprocess_arr(self._data["train_xs"])
        V = set(reduce(lambda l1, l2: l1 + l2, xs))
        print(len(V))
        # for x, y in zip(xs, self._data["train_ys"]):
        #     for token in x:

        return self
    
    def train(self):
        return self.build_log_prior().build_log_likelihood()

    def test(self):
        return

if __name__ == "__main__":
    p = Preprocessor()
    p.add_pipe(p.remove_digits).add_pipe(p.remove_not_alpha)

    nb = NaiveBayes(preprocessor=p)

    # load training data
    data = "./data/emr_data_10_icds.tsv"
    with open(data, encoding="utf-8") as f:
        df = pd.read_csv(f, sep="\t", index_col="index")
    train = df.groupby("icd").sample(frac=0.8)
    test = df.drop(train.index)

    nb.load(
        train_xs=train["text"].values,
        train_ys=train["icd"].values,
        test_xs=test["text"].values,
        test_ys=test["icd"].values
    )

    nb.train()