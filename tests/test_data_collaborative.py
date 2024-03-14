"""
    Test data realted functions, queries the DB for test data
    Unittest class for testing the DB read-in
    Rest in Pytest
"""

import pytest

import datetime

from tests.test_fixtures import inputs, testdata_collaborative

from src.data_content import get_data


def test_dataset_length(testdata_collaborative, limit=40000):
    """number of users and items is at least limit"""

    data_store_collaborative = testdata_collaborative

    n_users = data_store_collaborative.dataset.interactions_shape()[0]
    n_items = data_store_collaborative.dataset.interactions_shape()[1]

    assert n_users >= limit, f"dataset has n_users {n_users}, which is less than {limit}"
    assert n_items >= limit, f"dataset has n_items {n_users}, which is less than {limit}"
