"""
Hibrid retrieval pipeline
=========================
Architektúra:
  1. BM25          → top-N jelölt (pontos token egyezés, DN méretek, márkanevek)
  2. Bi-Encoder    → top-N jelölt (szemantikai hasonlóság)
  3. RRF Fusion    → két ranglista összefésülése
  4. Cross-Encoder → végső re-ranking (opcionális, pontosabb de lassabb)
 
Telepítés: csak standard könyvtárak + sentence-transformers
  pip install sentence-transformers scikit-learn numpy pandas
"""
from collections import defaultdict
import math
import numpy as np

class BM25:

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus: list[list[str]] = []
        self.doc_freqs: dict[str, int] = defaultdict(int)
        self.idoc_freqs: dict[str, float] = {}
        self.avgdl: float = 0.0
        self.N: int = 0

    def fit(self, tokenized_corpus: list[list[str]]):
        self.corpus = tokenized_corpus
        self.N = len(tokenized_corpus)
        self.avgdl = sum(len(doc) for doc in tokenized_corpus) / max(self.N, 1)

        #Dokumentum gyakoriság
        for doc in tokenized_corpus:
            for term in set(doc):
                self.doc_freqs[term] += 1
        
        #Inverz dokumentum gyakoriság
        for term, freq in self.doc_freqs.items():
            self.idoc_freqs[term] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)
        
        return self

    def get_score(self, query_tokens: list[str]) -> float:
        scores = np.zeros(self.N)
        for term in query_tokens:
            if term not in self.idoc_freqs:
                continue
            idf = self.idoc_freqs[term]
            for i, doc in enumerate(self.corpus):
                tf = doc.count(term)
                if tf == 0:
                    continue
                dl = len(doc)
                norm = tf * (self.k1 + 1)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * (norm / denom)
        return scores