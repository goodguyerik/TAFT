from collections import defaultdict

import nltk
import regex as re

from datafc.syn.token import BASIC_TYPES


def char_count(value):
    char_to_count = defaultdict(lambda: 0)
    for char in value:
        char_to_count[f"CHAR COUNT {char}"] += 1
    return char_to_count


def type_count(value):
    type_to_count = defaultdict(lambda: 0)
    for basic_type in BASIC_TYPES:
        type_to_count[f"TYPE COUNT f{basic_type.name}"] = len(
            re.findall(basic_type.regex, value)
        )
    return type_to_count


def word_count(value):
    word_to_count = defaultdict(lambda: 0)
    for word in nltk.wordpunct_tokenize(value):
        word_to_count[f"WORD COUNT {word}"] += 1
    return word_to_count


def length(value):
    return {"LENGTH": len(value)}
