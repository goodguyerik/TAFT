import string
from functools import lru_cache
from typing import Tuple, List, Optional

import regex as re


class Token:
    def __init__(
        self, token_type: "TokenType", position: int, length: int, values=None
    ):
        if values is None:
            values = []
        self.token_type: TokenType = token_type
        self.position: int = position
        self.length: int = length
        self.values = values

    def __hash__(self):
        return hash(self.token_type) + hash(self.position) + hash(self.length)

    def __str__(self):
        return f"TokenData({self.token_type.name}, {self.position}, {self.length})"

    def __eq__(self, token_data: "Token") -> bool:
        if (
            self.token_type == token_data.token_type
            and self.length == token_data.length
        ):
            return True
        return False

    def is_matched(self, token_data: "Token") -> bool:
        return (
            self.token_type == token_data.token_type
            and self.length == token_data.length
        )

    def combine(self, token_data: "Token") -> Optional["Token"]:
        return Token(
            self.token_type, self.position, self.length, self.values + token_data.values
        )

    @staticmethod
    def get_pattern_by_level(str_value: str, level) -> List["Token"]:
        masked_data: List["Token"] = [Token(StartToken, -1, -1, [""])]

        while str_value:
            for regex_type in PATTERNS_BY_LEVEL[level]:
                match_result = re.match("^" + regex_type.regex, str_value)
                if match_result:
                    if level == 0:
                        masked_data.append(
                            Token(
                                regex_type,
                                -1,
                                len(match_result.group()),
                                [match_result.group()],
                            )
                        )
                    else:
                        masked_data.append(
                            Token(regex_type, -1, -1, [match_result.group()])
                        )
                    str_value = str_value[len(match_result.group()) :]
                    break
        return masked_data

    @staticmethod
    @lru_cache(maxsize=2048)
    def get_basic_pattern(str_value: str) -> List["Token"]:
        masked_data: List["Token"] = [Token(StartToken, -1, -1, [""])]
        while str_value:
            for regex_type in BASIC_TYPES:
                match_result = re.match("^" + regex_type.regex, str_value)
                if match_result:
                    masked_data.append(
                        Token(
                            regex_type,
                            -1,
                            len(match_result.group()),
                            [match_result.group()],
                        )
                    )
                    str_value = str_value[len(match_result.group()) :]
                    break
        return masked_data

    @staticmethod
    @lru_cache(maxsize=2048)
    def get_basic_tokens(str_value: str) -> List["TokenType"]:
        str_value = str_value.strip()
        masked_data = [StartToken] * len(str_value)
        for regex_type in BASIC_TYPES:
            for match in re.finditer(regex_type.regex, str_value):
                masked_data[match.start()] = regex_type
        masked_data = [x for x in masked_data if x is not None]
        return masked_data


class TokenType:
    def __init__(self, name: str, regex: str, super_types=None):
        super().__init__()
        if super_types is None:
            super_types = []
        self._name: str = name
        self._regex: str = regex
        self.super_types: List["TokenType"] = super_types

    def __eq__(self, other: "TokenType"):
        return self.regex == other.regex and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def match(self, str_value: str):
        match = re.match(self.regex, str_value)
        if match:
            return True
        return False

    @property
    def name(self) -> str:
        return self._name

    @property
    def regex(self) -> str:
        return self._regex

    @regex.setter
    def regex(self, value: str):
        self._regex = value

    @property
    def super_types(self) -> List["TokenType"]:
        return self._super_types

    @super_types.setter
    def super_types(self, value):
        self._super_types = value

    @staticmethod
    def create_union_type(atomic_types: List["TokenType"]):
        regex = "|".join([f"({atomic_type.regex})" for atomic_type in atomic_types])
        return TokenType("", regex)

    @staticmethod
    def find_atomic_type(str_value: str) -> "TokenType":
        for atomic_type in BASIC_TYPES:
            if atomic_type.match(str_value):
                return atomic_type
        raise ValueError(f"Unrecognized type of value {str_value}")

    @staticmethod
    def find_all_matches_of_type(token_type: "TokenType", str_value: str) -> List[str]:
        return [match.group(0) for match in re.findall(token_type.regex, str_value)]

    @staticmethod
    def find_all_indices_of_type(token_type: "TokenType", str_value: str) -> List[str]:
        return [match.span() for match in re.findall(token_type.regex, str_value)]

    @staticmethod
    def find_all_tokens_of_type(
        token_type: "TokenType", str_value: str
    ) -> List[Tuple[int, int, Token]]:
        return [
            (
                match.start(),
                match.end(),
                Token(token_type, index, match.end() - match.start(), [match.group(0)]),
            )
            for index, match in enumerate(re.finditer(token_type.regex, str_value))
        ]

    @staticmethod
    def find_all_tokens_of_types(
        token_types: List["TokenType"], str_value: str
    ) -> List[Tuple[int, int, Token]]:
        result = []
        for type_name in token_types:
            result.extend(TokenType.find_all_tokens_of_type(type_name, str_value))
        return result

    @staticmethod
    def find_all_matches(str_value: str) -> List[Tuple[int, int, Token]]:
        return TokenType.find_all_tokens_of_types(REGEX_TYPES, str_value)


class ConstantType(TokenType):
    def __init__(self, const_str: str):
        self.regex = re.escape(const_str)
        super().__init__(f"Constant({const_str})", self.regex)


Text = TokenType("", r"[^(),;.]+")
Alnumspace = TokenType("Text", r"[\p{L}\p{N}]+(\p{Z}+[\p{L}\p{N}]+)*")
Alphanum = TokenType("Alphanum", r"[\p{L}\p{N}]+")
Alphabet = TokenType("Alphabet", r"[\p{L}]+", [Alphanum])
Uppercase = TokenType("Uppercase", r"\p{Lu}+", [Alphabet])
Lowercase = TokenType("Lowercase", r"\p{Ll}+", [Alphabet])
Decimal = TokenType("Digit", r"((?<![\.\p{N}+])\p{N}+\.\p{N}+(?![.\p{N}]))", [Alphanum])
Digit = TokenType("Digit", r"[\p{N}]+", [Alphanum, Alnumspace, Decimal])
Number = TokenType("Number", r" ")
Whitespace = TokenType("Whitespace", r"\p{Z}+")
StartToken = TokenType("^", re.escape("^"))
EndToken = TokenType("$", re.escape("$"))

PATTERNS_BY_LEVEL = [
    [Lowercase, Uppercase, Decimal, Digit, Whitespace, StartToken, EndToken],
    [Lowercase, Uppercase, Decimal, Digit, Whitespace, StartToken, EndToken],
    [Alphabet, Decimal, Digit, Whitespace, StartToken, EndToken],
    [Alphanum, Whitespace, StartToken, EndToken],
    [Alnumspace, Whitespace, StartToken, EndToken],
    # [Text, StartToken, EndToken],
]

DELIMITERS = [Whitespace] + [ConstantType(x) for x in ".,;/!?@:"]
BASIC_TYPES = [Uppercase, Lowercase, Digit, Whitespace, StartToken, EndToken]
REGEX_TYPES = [
    Alphanum,
    Alphabet,
    Uppercase,
    Lowercase,
    Digit,
    Whitespace,
    StartToken,
    EndToken,
]

ALL_TYPES = [
    Alnumspace,
    Text,
    Alphanum,
    Alphabet,
    Uppercase,
    Lowercase,
    Decimal,
    Digit,
    Whitespace,
    StartToken,
    EndToken,
]

for mark in string.punctuation:
    BASIC_TYPES.append(TokenType(f"{mark}", re.escape(mark)))
    REGEX_TYPES.append(TokenType(f"{mark}", re.escape(mark)))
    for layer in range(0, len(PATTERNS_BY_LEVEL)):
        if layer == 5:
            if mark in ",;.()":
                PATTERNS_BY_LEVEL[layer].append(TokenType(f"{mark}", re.escape(mark)))
        else:
            PATTERNS_BY_LEVEL[layer].append(TokenType(f"{mark}", re.escape(mark)))
