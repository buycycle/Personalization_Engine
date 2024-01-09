"""
 test collaborative recommendation provision realted functions
"""

import time
import random


from src.strategies import CollaborativeRandomized


def test_time_CollaborativeRandomized(inputs, testdata_collaborative, testdata_content, limit=100):
    """test time of recommendation, limit in ms"""
    bike_id, distinct_id, family_id, price, frame_size_code, n, sample, ratio, client, logger = inputs

    data_store_collaborative = testdata_collaborative
    data_store_content = testdata_content

    # convert from ms to s
    limit = limit / 1000

    start_time = time.time()

    collaborative_strategy = CollaborativeRandomized(logger, data_store_collaborative, data_store_content)

    strategy, recommendation, error = collaborative_strategy.get_recommendations(distinct_id, n, sample)

    end_time = time.time()
    assert (
        end_time - start_time < limit
    ), f"CollaborativeRandomized took {(end_time - start_time)*1000} ms, limit is {limit*1000} ms"


def test_len_CollaborativeRandomized(inputs, testdata_collaborative, testdata_content, n_test=100):
    """test length of recommendation list for a random subset of user in dataset"""
    bike_id, distinct_id, family_id, price, frame_size_code, n, sample, ratio, client, logger = inputs

    data_store_collaborative = testdata_collaborative
    data_store_content = testdata_content

    collaborative_strategy = CollaborativeRandomized(logger, data_store_collaborative, data_store_content)

    users = data_store_collaborative.dataset.mapping()[0].keys()
    users = list(users)

    # subsample
    for i in random.sample(users, n_test):
        distinct_id = i

        strategy, recommendation, error = collaborative_strategy.get_recommendations(distinct_id, n, sample)

        assert (
            len(recommendation) == n
        ), f"recommendation has {len(recommendation)} rows for distinct_id {distinct_id}, which is not {n} rows"
