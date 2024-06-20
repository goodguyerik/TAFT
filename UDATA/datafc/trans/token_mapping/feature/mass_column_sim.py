from typing import List

from datasketch import MinHashLSH

from datafc.repr.column import Column


def mass_values_jaccard(cols1: List[Column], cols2: List[Column]):
    lsh = MinHashLSH(
        threshold=0.2,
        num_perm=128,
        storage_config={"type": "redis", "redis": {"host": "localhost", "port": 6379}},
    )

    with lsh.insertion_session() as session:
        for idx, col in enumerate(cols1):
            session.insert(str(idx), col.values)

    result = lsh.query()
