"""
 test content recommendation provision realted functions
 test with prepacked test DB read-in, as fixtures
"""

import time

from tests.test_fixtures import inputs, testdata_content, testdata_collaborative

import numpy as np
from src.driver_content import prefilter_features

from src.strategies import ContentMixed

from buycycle.data import get_preference_mask


def test_time_ContentMixed(inputs, testdata_collaborative, testdata_content, limit=100):
    """test time of recommendation for a predifined bike_id with DB read-in"""
    (
        bike_id,
        preferences,
        bike_type,
        distinct_id,
        family_id,
        price,
        frame_size_code,
        n,
        sample,
        ratio,
        app,
        logger,
    ) = inputs

    data_store_collaborative = testdata_collaborative
    data_store_content = testdata_content
    preference_mask = get_preference_mask(data_store_content.df_preference, preferences)

    # s to ms
    limit = limit / 1000

    start_time = time.time()

    mixed_strategy = ContentMixed(logger, data_store_collaborative, data_store_content)
    strategy, recommendation, error = mixed_strategy.get_recommendations(
        bike_id, preference_mask, bike_type, family_id, price, frame_size_code, n
    )

    end_time = time.time()
    assert (
        end_time - start_time < limit
    ), f"ContentMixed took {(end_time - start_time)*1000} ms, limit is {limit*1000} ms"


def test_len_random_ContentMixed(
    inputs, testdata_collaborative, testdata_content, n_test=100
):
    """test length of recommendation list for radnom bike ids
    similarity_matrix rows are bike_ids with all statuses
    """

    (
        bike_id,
        preferences,
        bike_type,
        distinct_id,
        family_id,
        price,
        frame_size_code,
        n,
        sample,
        ratio,
        app,
        logger,
    ) = inputs

    data_store_collaborative = testdata_collaborative
    data_store_content = testdata_content
    preference_mask = get_preference_mask(data_store_content.df_preference, preferences)

    # do n_test times for i between 0 and 50000
    for i in range(n_test):
        # i is a random number between 0 and 50000
        bike_id = int(50000 * np.random.random_sample())

        mixed_strategy = ContentMixed(
            logger, data_store_collaborative, data_store_content
        )
        strategy, recommendation, error = mixed_strategy.get_recommendations(
            bike_id, preferences, bike_type, family_id, price, frame_size_code, n
        )

        assert (
            len(recommendation) == n
        ), f"ContentMixed recommendation has {len(recommendation)} rows for bike_id {bike_id}, which is not {n} rows"
