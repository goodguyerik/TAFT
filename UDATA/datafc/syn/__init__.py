from typing import Dict, List

from datafc.syn.token import TokenType, TokenType

type_name_to_regex: Dict[str, str] = {
    "Uppercase": r"\p{Lu}+",
    "Lowercase": r"\p{Ll}+",
    "Titlecase": r"\p{Lt}+",
    "Digit": r"\p{N}+",
    "Alphabet": r"\p{L}+",
    "Alphanum": r"[\p{L}\p{N}]+",
    "Whitespace": r"\p{Z}+",
}

accepted_types: List[TokenType] = []

for type_name, regex in type_name_to_regex.items():
    accepted_types.append(TokenType(type_name, regex))
