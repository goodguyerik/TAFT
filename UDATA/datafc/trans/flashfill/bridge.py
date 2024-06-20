from typing import List

from py4j.java_gateway import JavaGateway


class FlashFillBridge:
    def __init__(self, gateway: JavaGateway):
        self.gateway = gateway

    def learn(
        self,
        input_strings: List[str],
        output_strings: List[str],
        test_strings: List[str],
    ):
        return self.gateway.entry_point.programSynthesize(
            input_strings, output_strings, test_strings
        )
