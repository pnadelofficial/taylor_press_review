# rewrote the earlier code into a simple class
from collections import Counter
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

class TFIDF:
    def __init__(self, texts):
        self.texts = texts
        self.tokenized_texts = [self.tokenize(text) for text in texts]
        self.tfidf = self.calc_tfidf()

    def tokenize(self, text):
        text = text.lower()
        words = word_tokenize(text)
        words = [word.lower() for word in words if word.isalpha()] 
        return words

    def count(self):
        tokens = pd.Series(self.tokenized_texts)
        counts = tokens.apply(Counter)
        return counts.apply(pd.Series).fillna(0).astype(int)

    def calc_tfidf(self):
        dtm = self.count()
        tf = dtm.div(dtm.sum(axis=1), axis=0)

        n_documents = len(dtm)
        self.idf = np.log(n_documents/(dtm > 0).sum(axis=0))

        tfidf = tf * self.idf
        return tfidf

    def query(self, query):
        query_tokens = self.tokenize(query)
        query_counts = Counter(query_tokens)
        query_vector = pd.Series(0, index=self.tfidf.columns)
        for term, count in query_counts.items():
            if term in self.tfidf.columns:
                query_vector[term] = count
        query_tf = query_vector / sum(query_vector) if sum(query_vector) > 0 else query_vector
        query_tfidf = query_tf * self.idf

        sim_scores = query_tfidf.to_numpy() @ self.tfidf.to_numpy().T
        return list(zip(sim_scores.argsort()[::-1], sim_scores[sim_scores.argsort()[::-1]]))
