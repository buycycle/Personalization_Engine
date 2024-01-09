from typing import Any
import pandas as pd


def interveave(list1: list, list2: list) -> list:
    """interveave two lists"""

    return [item for x in zip(list1, list2) for item in x] + (
        list2[len(list1) :] if len(list2) > len(list1) else list1[len(list2) :]
    )
