"""
functional tests for the fastapi model response
"""

import time

import random

import numpy as np
from src.driver_content import prefilter_features

from src.strategies import ContentMixed


def test_integration_fast_time_len_strats_input(inputs_fastapi, limit=100):
    """test time and len of return for all strategies of the fastapi app"""
    bike_id, distinct_id, family_id, price, frame_size_code, n, sample, ratio, client, logger, strategy_dict = inputs_fastapi
    strategies = list(strategy_dict.keys())

    # s to ms
    limit = limit / 1000

    # exclude braze and homepage since they do not ensure returning n
    to_remove = ["braze", "homepage"]
    strategies = [item for item in strategies if item not in to_remove]

    for strategy in strategies:
        # prepare the request payload with the current strategy
        payload = {
            "bike_id": bike_id,
            "distinct_id": distinct_id,
            "family_id": family_id,
            "price": price,
            "frame_size_code": frame_size_code,
            "n": n,
            "strategy": strategy,  # set the current strategy
        }
        # simulate a post request to the recommendation endpoint
        start_time = time.time()
        response = client.post("/recommendation", json=payload)
        end_time = time.time()

        # ensure the request was successful
        assert response.status_code == 200, f"request failed with status code {response.status_code} for strategy {strategy}"
        # parse the response data
        data = response.json()
        recommendation = data.get("recommendation")
        strategy_used = data.get("strategy")
        # check the time taken for the recommendation
        assert end_time - start_time < limit, f"{strategy} took {(end_time - start_time)*1000} ms, limit is {limit*1000} ms"
        # assert that the response has the expected length n
        assert len(recommendation) == n, f"expected {n} recommendations for strategy {strategy}, got {len(recommendation)}"
        # ... (other assertions or checks based on the response)


def test_integration_fast_time_len_strats_collab_users(inputs_fastapi, testdata_collaborative, limit=100, n_test=10):
    """test time and len of return for all strategies and a random subsample of collaborative users of the fastapi app"""
    bike_id, distinct_id, family_id, price, frame_size_code, n, sample, ratio, client, logger, strategy_dict = inputs_fastapi
    strategies = list(strategy_dict.keys())
    data_store_collaborative = testdata_collaborative

    users = data_store_collaborative.dataset.mapping()[0].keys()
    users = list(users)

    # exclude braze and homepage since they do not ensure returning n
    to_remove = ["braze", "homepage"]
    # Remove items from strategies if they exist
    strategies = [item for item in strategies if item not in to_remove]

    # subsample
    for i in random.sample(users, n_test):
        distinct_id = i

        for strategy in strategies:
            # prepare the request payload with the current strategy
            payload = {
                "bike_id": bike_id,
                "distinct_id": distinct_id,
                "family_id": family_id,
                "price": price,
                "frame_size_code": frame_size_code,
                "n": n,
                "strategy": strategy,  # set the current strategy
            }
            # simulate a post request to the recommendation endpoint
            start_time = time.time()
            response = client.post("/recommendation", json=payload)
            end_time = time.time()

            # ensure the request was successful
            assert response.status_code == 200, f"request failed with status code {response.status_code} for strategy {strategy}"
            # parse the response data
            data = response.json()
            recommendation = data.get("recommendation")
            strategy_used = data.get("strategy")
            # Check if the strategy is 'homepage' and verify the corresponding strategy
            if strategy == 'homepage':
                expected_strategy = strategy_dict[strategy].__name__  # Get the class name as a string
                assert strategy_used == expected_strategy, f"expected strategy {expected_strategy}, got {strategy_used} for 'product_page'"
            # check the time taken for the recommendation
            assert end_time - start_time < limit, f"{strategy} took {(end_time - start_time)*1000} ms, limit is {limit*1000} ms"
            # assert that the response has the expected length n
            assert len(recommendation) == n, f"expected {n} recommendations for strategy {strategy}, got {len(recommendation)}"
            # ... (other assertions or checks based on the response)


def test_integration_fast_time_len_strats_bikes(inputs_fastapi, limit=150, n_test=100):
    """test time and len of return for all strategies and a random bike_ids of the fastapi app"""
    bike_id, distinct_id, family_id, price, frame_size_code, n, sample, ratio, client, logger, strategy_dict = inputs_fastapi
    strategies = list(strategy_dict.keys())

    # exclude braze and homepage since they do not ensure returning n
    to_remove = ["braze", "homepage"]
    # Remove items from strategies if they exist
    strategies = [item for item in strategies if item not in to_remove]

    random_bike_ids = np.random.choice(50000, size=n_test, replace=False).tolist()

    for bike_id in random_bike_ids:
        for strategy in strategies:
            # prepare the request payload with the current strategy
            payload = {
                "bike_id": bike_id,
                "distinct_id": distinct_id,
                "family_id": family_id,
                "price": price,
                "frame_size_code": frame_size_code,
                "n": n,
                "strategy": strategy,  # set the current strategy
            }
            # simulate a post request to the recommendation endpoint
            start_time = time.time()
            response = client.post("/recommendation", json=payload)
            end_time = time.time()

            # ensure the request was successful
            assert response.status_code == 200, f"request failed with status code {response.status_code} for strategy {strategy}"
            # parse the response data
            data = response.json()
            recommendation = data.get("recommendation")
            strategy_used = data.get("strategy")
            # Check if the strategy is 'product_page' and verify the corresponding strategy
            if strategy == 'product_page':
                expected_strategy = strategy_dict[strategy].__name__  # Get the class name as a string
                assert strategy_used == expected_strategy, f"expected strategy {expected_strategy}, got {strategy_used} for 'product_page'"
            # check the time taken for the recommendation
            assert end_time - start_time < limit, f"{strategy} took {(end_time - start_time)*1000} ms, limit is {limit*1000} ms"
            # assert that the response has the expected length n
            assert len(recommendation) == n, f"expected {n} recommendations for strategy {strategy}, got {len(recommendation)}"
            # ... (other assertions or checks based on the response)
