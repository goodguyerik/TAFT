from abc import abstractmethod, ABCMeta
from typing import List

from datafc.syn.token import Token, Uppercase, Lowercase, StartToken


class Operation(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, original_token: Token, target_token: Token, **kwargs):
        self.original_token = original_token
        self.target_token = target_token

    @abstractmethod
    def transform(self) -> List[str]:
        pass

    @staticmethod
    def find_suitable_transformations(
        original_token: Token, target_token: Token
    ) -> List["Operation"]:
        transform_ops: List["Operation"] = []
        for Op in [Upper, Lower, Replace, Constant, Substring]:
            transform_ops.extend(Op.generate(original_token, target_token))
        return transform_ops


class Substring(Operation):
    def __init__(
        self, original_token: Token, target_token: Token, start_index: int, length: int
    ):
        super(Substring, self).__init__(original_token, target_token)
        self.start_index: int = start_index
        self.length: int = length

    def __str__(self):
        return f"Substring({self.start_index},{self.length})"

    @staticmethod
    def find_all_parameters(original_token: Token, target_token: Token):
        is_same_length = len(set([len(x) for x in original_token.values])) == 1

        original_min_length = min([len(x) for x in original_token.values])
        target_length = len(target_token.values[0])

        params_list = []

        for i in range(original_min_length - target_length + 1):
            params_list.append((i, target_length))
            if not is_same_length:
                params_list.append((-i - target_length, target_length))

        return params_list

    def transform(self) -> List[str]:
        if self.length == -1:
            return [x[self.start_index :] for x in self.original_token.values]
        return [
            x[self.start_index : self.start_index + self.length]
            for x in self.original_token.values
        ]

    @staticmethod
    def generate(original_token: Token, target_token: Token) -> List["Operation"]:
        if (
            target_token.token_type == StartToken
            or len(set([len(x) for x in target_token.values])) != 1
        ):
            return []

        params_list = Substring.find_all_parameters(original_token, target_token)

        return [
            Substring(original_token, target_token, x[0], x[1]) for x in params_list
        ]


class Upper(Operation):
    def __init__(self, original_token: Token, target_token: Token):
        super(Upper, self).__init__(original_token, target_token)

    def __str__(self):
        return "Upper"

    def transform(self) -> List[str]:
        return [x.upper() for x in self.original_token.values]

    @staticmethod
    def generate(original_token: Token, target_token: Token) -> List["Operation"]:
        if (
            target_token.token_type != StartToken
            and original_token.token_type != Uppercase
            and target_token.token_type == Uppercase
        ):
            return [Upper(original_token, target_token)]
        return []


class Lower(Operation):
    def __init__(self, original_token: Token, target_token: Token):
        super(Lower, self).__init__(original_token, target_token)

    def __str__(self):
        return "Lower"

    def transform(self) -> List[str]:
        return [x.lower() for x in self.original_token.values]

    @staticmethod
    def generate(original_token: Token, target_token: Token) -> List["Operation"]:
        if (
            target_token.token_type != StartToken
            and original_token.token_type != Lowercase
            and target_token.token_type == Lowercase
        ):
            return [Lower(original_token, target_token)]
        return []


class PartialReplace(Operation):
    def __init__(
        self, original_token: Token, target_token: Token, start_index: int, length: int
    ):
        self.start_index = start_index
        self.length = length
        super(PartialReplace, self).__init__(original_token, target_token)

    def __str__(self):
        return f"PartialReplace({self.start_index}, {self.length})"

    def transform(self) -> List[str]:
        return [
            self.target_token.values[0][: self.start_index]
            + value
            + self.target_token.values[0][self.start_index + self.length :]
            for value in self.original_token.values
        ]

    @staticmethod
    def find_all_parameters(original_token, target_token):
        is_same_length = len(set([len(x) for x in target_token.values])) == 1

        target_min_length = min([len(x) for x in target_token.values])
        original_length = len(original_token.values[0])

        params_list = []

        for i in range(target_min_length - original_length + 1):
            if (
                len(
                    set([x[:i] + x[i + original_length :] for x in target_token.values])
                )
                == 1
            ):
                params_list.append((i, original_length))
            if not is_same_length:
                if (
                    len(
                        set(
                            [
                                x[: -i - original_length] + x[-i:]
                                for x in target_token.values
                            ]
                        )
                    )
                    == 1
                ):
                    params_list.append((-i - original_length, original_length))

        return params_list

    @staticmethod
    def generate(original_token: Token, target_token: Token) -> List["Operation"]:
        if (
            target_token.token_type == StartToken
            or len(set([len(x) for x in original_token.values])) != 1
        ):
            return []
        params_list = PartialReplace.find_all_parameters(original_token, target_token)
        return [
            PartialReplace(original_token, target_token, x[0], x[1])
            for x in params_list
        ]


class Replace(Operation):
    def __init__(self, original_token: Token, target_token: Token):
        super(Replace, self).__init__(original_token, target_token)

    def __str__(self):
        return "Replace"

    def transform(self) -> List[str]:
        return self.original_token.values

    @staticmethod
    def generate(original_token: Token, target_token: Token) -> List["Operation"]:
        if (
            target_token.token_type != StartToken
            and target_token.token_type == original_token.token_type
        ):
            if (
                len(
                    set([len(x) for x in target_token.values]).intersection(
                        set([len(x) for x in original_token.values])
                    )
                )
                != 0
            ):
                return [Replace(original_token, target_token)]
        return []


class Constant(Operation):
    def __init__(self, original_token: Token, target_token: Token):
        super(Constant, self).__init__(original_token, target_token)

    def __str__(self):
        return f"Constant({self.target_token.values[0]})"

    def transform(self) -> List[str]:
        return [
            self.target_token.values[0] for _ in range(len(self.original_token.values))
        ]

    @staticmethod
    def generate(original_token: Token, target_token: Token) -> List["Operation"]:
        if target_token.token_type == StartToken:
            return [Constant(original_token, target_token)]
        if len(set(target_token.values)) == 1 and len(target_token.values) != 1:
            return [Constant(original_token, target_token)]
        return []
