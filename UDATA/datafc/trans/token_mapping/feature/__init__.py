import locale
import logging
from typing import Optional

import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("myapp")


def jaccard(x, y):
    if len(x) > 0 and len(y) > 0:
        x = set(x)
        y = set(y)
        return float(len(x.intersection(y))) / float(len(x.union(y)))
    return 0


def cosine(x: str, y: str):
    vectorizer = CountVectorizer()
    try:
        vectors = vectorizer.fit_transform([x, y]).toarray()
    except Exception as e:
        logger.error(e)
        return 0
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]


def to_number(num_str: str) -> Optional[float]:
    locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
    try:
        return locale.atof(num_str)
    except ValueError:
        return None


# class Spacy:
#     model = spacy.load("en_core_web_lg")
