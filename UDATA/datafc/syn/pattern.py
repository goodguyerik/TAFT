from functools import reduce
from typing import List, Dict, Optional

import regex as re

from datafc.repr.column import Column
from datafc.syn.token import (
    Token,
    Uppercase,
    Lowercase,
    Alphabet,
    Digit,
    Alphanum,
    Whitespace,
    Alnumspace,
    PATTERNS_BY_LEVEL,
)


class Pattern:
    def __init__(self, token_sequence: List[Token], level=0):
        self.tokens: List[Token] = token_sequence
        self.level = level
        self.values = ["" for _ in range(len(self.tokens[0].values))]

        self.regex = "".join([token.token_type.regex for token in self.tokens])

        for token in self.tokens:
            for idx in range(len(token.values)):
                self.values[idx] += token.values[idx]

    def __repr__(self):
        return f"Pattern({''.join([str(token) for token in self.tokens])}, level={self.level})"

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, pattern: "Pattern") -> bool:
        if len(self.tokens) != len(pattern.tokens):
            return False
        for idx in range(len(self.tokens)):
            if self.tokens[idx] != pattern.tokens[idx]:
                return False
        return True

    def combine(self, pattern: "Pattern") -> "Pattern":
        assert self == pattern
        new_tokens = []
        for idx, token in enumerate(self.tokens):
            new_tokens.append(token.combine(pattern.tokens[idx]))

        return Pattern(new_tokens, self.level)

    def match(self, value_str: str) -> bool:
        if re.fullmatch(self.regex, value_str):
            return True
        return False

    def match_to_pattern(self, value_str: str) -> Optional["Pattern"]:
        tokens = []
        for token in self.tokens:
            match_result = re.match(r"^%s" % token.token_type.regex, value_str)
            if match_result:
                tokens.append(
                    Token(
                        token.token_type,
                        token.position,
                        token.length,
                        values=[match_result.group()],
                    )
                )
                value_str = value_str[len(match_result.group()) :]
            else:
                return None
        return Pattern(tokens, self.level)

    def to_cols(self):
        columns = []

        for idx, token in enumerate(self.tokens):
            column = Column(str(idx), str(idx), token.values)
            columns.append(column)

        return columns

    @staticmethod
    def build_from_string(string: str, level) -> "Pattern":
        tokens = Token.get_pattern_by_level(string, level)

        return Pattern(tokens, level=level)

    @staticmethod
    def build_and_match_from_string(
        string: str, level: int, patterns: List["PatternNode"]
    ) -> "Pattern":
        for pattern_node in patterns:
            match_result = pattern_node.value.match_to_pattern(string)
            if match_result:
                return match_result
        else:
            tokens = Token.get_pattern_by_level(string, level)

        return Pattern(tokens, level=level)

    def level_up(self) -> Optional["Pattern"]:
        tokens: List[Optional[Token]] = [None] * len(self.tokens)
        for idx, token in enumerate(self.tokens):
            if self.level == 0:
                tokens[idx] = Token(token.token_type, -1, -1, token.values)
            elif self.level == 1:
                if token.token_type == Uppercase or token.token_type == Lowercase:
                    tokens[idx] = Token(Alphabet, -1, -1, token.values)
                else:
                    tokens[idx] = self.tokens[idx]
            elif self.level == 2:
                if token.token_type == Alphabet or token.token_type == Digit:
                    tokens[idx] = Token(Alphanum, -1, -1, token.values)
                else:
                    tokens[idx] = self.tokens[idx]
            elif self.level == 3:
                if token.token_type == Alphanum or token.token_type == Whitespace:
                    tokens[idx] = Token(Alnumspace, -1, -1, token.values)
                else:
                    tokens[idx] = self.tokens[idx]

        idx = 0
        removed_indices = []
        while idx < len(tokens):
            run_idx = idx + 1
            while run_idx < len(tokens):
                if tokens[run_idx].token_type == tokens[idx].token_type:
                    tokens[idx].values = [
                        value + tokens[run_idx].values[i]
                        for i, value in enumerate(tokens[idx].values)
                    ]
                    removed_indices.append(run_idx)
                    run_idx += 1
                else:
                    break
            idx = run_idx

        for remove_idx in reversed(removed_indices):
            del tokens[remove_idx]

        if tokens:
            return Pattern(tokens, self.level + 1)
        return None


class PatternNode:
    def __init__(self, value: Pattern, parent: "PatternNode" = None, children=None):
        if children is None:
            children = []
        self.value: Pattern = value
        self.children: List["PatternNode"] = children
        self.parent = parent

    def __repr__(self):
        return f"Node({self.value}, level={self.value.level})"


class PatternTree:
    def __init__(self, values: List[str]):
        self.node_in_layers: List[List[PatternNode]] = [
            [] for _ in range(0, len(PATTERNS_BY_LEVEL))
        ]
        self.pattern_to_node: Dict[Pattern, PatternNode] = {}
        self.values = values

    def get_all_patterns(self):
        return reduce(list.__add__, self.node_in_layers, [])

    def get_patterns_by_layers(self, layers, in_groups=False):
        if in_groups:
            return [
                [node.value for node in self.node_in_layers[layer]] for layer in layers
            ]
        return [
            pattern_node.value
            for pattern_node in reduce(
                list.__add__, [self.node_in_layers[layer] for layer in layers], []
            )
        ]

    @staticmethod
    def build_from_strings(str_values: List[str]) -> "PatternTree":
        pattern_to_parent: Dict[Pattern, Pattern] = {}
        pattern_tree = PatternTree(str_values)

        for str_value in str_values:
            child_pattern = None
            for layer in range(1, len(PATTERNS_BY_LEVEL)):
                if child_pattern in pattern_to_parent:
                    pattern = pattern_to_parent[child_pattern]
                else:
                    pattern = Pattern.build_and_match_from_string(
                        str_value.strip(), layer, pattern_tree.node_in_layers[layer]
                    )

                if pattern in pattern_tree.pattern_to_node:
                    pattern_tree.pattern_to_node[
                        pattern
                    ].value = pattern_tree.pattern_to_node[pattern].value.combine(
                        pattern
                    )
                else:
                    pattern_node = PatternNode(pattern)
                    pattern_tree.pattern_to_node[pattern] = pattern_node
                    pattern_tree.node_in_layers[layer].append(pattern_node)
                child_pattern = pattern
        return pattern_tree

    # @staticmethod
    # def build_from_strings(str_values: List[str]) -> "PatternTree":
    #     pattern_tree = PatternTree(str_values)
    #     pattern_to_node: Dict[str, PatternNode] = {}
    #
    #     for str_value in str_values:
    #         pattern = Pattern.build_from_string(str_value.strip())
    #         if str(pattern) in pattern_to_node:
    #             pattern_to_node[str(pattern)].value = pattern.combine(pattern_to_node[str(pattern)].value)
    #         else:
    #             pattern_tree.node_in_layers[0].append(PatternNode(pattern))
    #
    #     pattern_to_node.clear()
    #     for i in range(1, 5):
    #         for pattern_node in pattern_tree.node_in_layers[i - 1]:
    #             parent_pattern = pattern_node.value.level_up()
    #
    #             if parent_pattern is None:
    #                 continue
    #             if str(parent_pattern) in pattern_to_node:
    #                 pattern_to_node[str(parent_pattern)].value = pattern_to_node[str(parent_pattern)].value.combine(
    #                     parent_pattern
    #                 )
    #                 pattern_to_node[str(parent_pattern)].children.append(pattern_node)
    #                 pattern_node.parent = pattern_to_node[str(parent_pattern)]
    #             else:
    #                 parent_node = PatternNode(parent_pattern)
    #                 parent_node.children.append(pattern_node)
    #                 pattern_node.parent = parent_node
    #                 pattern_tree.node_in_layers[i].append(parent_node)
    #                 pattern_to_node[str(parent_pattern)] = parent_node
    #         pattern_to_node.clear()
    #     return pattern_tree
