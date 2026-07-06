# rewrote the earlier code into a simple class
import html
import re
from collections import Counter
from functools import lru_cache
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk

nltk.download('punkt')


@lru_cache(maxsize=256)
def _query_pattern(query):
    """Compile a case-insensitive, whole-word regex for the searchable terms in
    ``query``. Returns ``None`` when the query has no alphabetic terms.

    Cached because ``search_snippet`` is called once per article row while the
    query is constant across the loop.
    """
    terms = [word.lower() for word in word_tokenize(query) if word.isalpha()]
    if not terms:
        return None
    return re.compile(
        r"\b(" + "|".join(re.escape(term) for term in terms) + r")\b",
        re.IGNORECASE,
    )


def _escape(value):
    """Escape ``value`` for safe display via ``st.markdown(unsafe_allow_html)``.

    Beyond HTML-escaping ``&<>``, ``$`` is turned into the ``&#36;`` entity so
    Streamlit's markdown doesn't render ``$...$`` spans as LaTeX.
    """
    return html.escape(value).replace("$", "&#36;")


def _highlight(text, pattern):
    """Escape ``text`` and wrap ``pattern`` matches in ``<mark>`` tags.

    Escaping is done segment by segment on the *raw* text so the injected
    ``<mark>`` tags are the only unescaped HTML in the result — matches can
    never land inside an escaped entity (e.g. the ``amp`` in ``&amp;``).
    """
    out = []
    last = 0
    for match in pattern.finditer(text):
        out.append(_escape(text[last:match.start()]))
        out.append("<mark>" + _escape(match.group(0)) + "</mark>")
        last = match.end()
    out.append(_escape(text[last:]))
    return "".join(out)


def search_snippet(text, query, context_size=1):
    """Return a short context window around the first match of ``query`` in
    ``text`` with the matched term(s) highlighted.

    Rather than the whole article, this returns the sentence containing the
    first match plus ``context_size`` sentences before and after it. The text is
    HTML-escaped and matched terms are wrapped in ``<mark>`` tags, so render the
    result with ``st.markdown(..., unsafe_allow_html=True)``. Because the source
    text is untrusted (scraped article bodies), escaping the raw text is what
    keeps stray ``<...>`` sequences and any embedded markup from being
    interpreted as HTML.

    Falls back to the full (escaped) text if the query has no searchable terms
    or no match is found.
    """
    text = str(text)
    pattern = _query_pattern(query)

    if pattern is None:
        return _escape(text)

    sentences = sent_tokenize(text)
    for i, sentence in enumerate(sentences):
        if pattern.search(sentence):
            start = max(0, i - context_size)
            end = i + context_size + 1
            snippet = " ".join(sentences[start:end])
            return _highlight(snippet, pattern)

    # no sentence matched (e.g. term only in metadata) -> show everything
    return _escape(text)

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
