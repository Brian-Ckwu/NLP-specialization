from week2_naive_bayes.multiclass_nb.preprocessor import Preprocessor


from preprocessor import Preprocessor

class NaiveBayes(object):
    def __init__(self, preprocessor: Preprocessor):
        self._preprocessor = preprocessor