from typing import List, Set
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from string import punctuation

class Preprocessor(object):
    def __init__(self, stopwords: Set[str] = set(stopwords.words("english")), punc: Set[str] = set(punctuation)):
        self._stopwords = stopwords
        self._punc = punc
    
    @property
    def stopwords(self) -> Set[str]:
        return self._stopwords

    @property
    def punctuations(self) -> Set[str]:
        return self._punc

    def tokenize(self, text: str) -> List[str]:
        return wordpunct_tokenize(text)
    
    def remove_sw_punc(self, tokens: List[str]) -> List[str]:
        clean_tokens = list()
        for token in tokens:
            if (token not in self._stopwords) and (token not in self._punc):
                clean_tokens.append(token)
        return clean_tokens
    
    def to_lowercase(self, tokens: List[str]) -> List[str]:
        lowered = list()
        for token in tokens:
            lowered.append(token.lower())
        return lowered
    
    def remove_digits(self, tokens: List[str]) -> List[str]:
        return list(filter(lambda t: not t.isdigit(), tokens))
    
    def remove_not_alpha(self, tokens: List[str]) -> List[str]:
        return list(filter(lambda t: t.isalpha(), tokens))
    
    def preprocess(self, text: str, added_pipes: list = []) -> List[str]:
        tokens = self.tokenize(text)
        c_tokens = self.remove_sw_punc(tokens)
        l_tokens = self.to_lowercase(c_tokens)

        p_tokens = l_tokens
        for func in added_pipes:
            p_tokens = func(p_tokens)
        
        return p_tokens

if __name__ == "__main__":
    import pandas as pd
    p = Preprocessor()
    
    file = "./data/emr_data_10_icds.tsv"
    with open(file, encoding="utf-8") as f:
        df = pd.read_csv(f, sep='\t')
    
    idx = 10
    text = df.at[idx, "text"]
    print(text)
    print(p.preprocess(text, added_pipes=[p.remove_digits, p.remove_not_alpha]))