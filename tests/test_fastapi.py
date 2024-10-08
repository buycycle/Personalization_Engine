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
    (
        bike_id,
        continent_id,
        bike_type,
        category,
        distinct_id,
        family_id,
        price,
        frame_size_code,
        rider_height_min,
        rider_height_max,
        rider_height,
        n,
        sample,
        ratio,
        client,
        logger,
        strategy_dict,
    ) = inputs_fastapi
    strategies = list(strategy_dict.keys())

    # s to ms
    limit = limit / 1000

    # exclude braze and homepage since they do not ensure returning n
    to_remove = ["braze", "homepage", "bot"]
    strategies = [item for item in strategies if item not in to_remove]

    for strategy in strategies:
        # prepare the request payload with the current strategy
        payload = {
            "bike_id": bike_id,
            "continent_id": continent_id,
            "bike_type": bike_type,
            "category": category,
            "distinct_id": distinct_id,
            "family_id": family_id,
            "price": price,
            "frame_size_code": frame_size_code,
            "rider_height": rider_height,
            "n": n,
            "strategy": strategy,  # set the current strategy
        }
        # simulate a post request to the recommendation endpoint
        start_time = time.time()
        response = client.post("/recommendation", json=payload)
        end_time = time.time()

        # ensure the request was successful
        assert (
            response.status_code == 200
        ), f"request failed with status code {response.status_code} for strategy {strategy}"
        # parse the response data
        data = response.json()
        recommendation = data.get("recommendation")
        strategy_used = data.get("strategy")
        # check the time taken for the recommendation
        assert (
            end_time - start_time < limit
        ), f"{strategy} took {(end_time - start_time)*1000} ms, limit is {limit*1000} ms"
        # assert that the response has the expected length n
        assert (
            len(recommendation) == n
        ), f"expected {n} recommendations for strategy {strategy}, got {len(recommendation)}"
        assert all(isinstance(item, int) for item in recommendation), \
            f"expected all recommendations to be integers for strategy {strategy}, got {recommendation}"
        # ... (other assertions or checks based on the response)


def test_integration_fast_time_len_strats_collab_users(
    inputs_fastapi, testdata_collaborative, limit=50, n_test=10
):
    """test time and len of return for all strategies and a random subsample of collaborative users of the fastapi app"""
    (
        bike_id,
        continent_id,
        bike_type,
        category,
        distinct_id,
        family_id,
        price,
        frame_size_code,
        rider_height_min,
        rider_height_max,
        rider_height,
        n,
        sample,
        ratio,
        client,
        logger,
        strategy_dict,
    ) = inputs_fastapi
    strategies = list(strategy_dict.keys())
    data_store_collaborative = testdata_collaborative

    users = data_store_collaborative.dataset.mapping()[0].keys()
    users = list(users)

    # exclude braze and homepage since they do not ensure returning n
    to_remove = ["braze", "homepage", "bot"]
    # Remove items from strategies if they exist
    strategies = [item for item in strategies if item not in to_remove]

    # subsample
    for i in random.sample(users, n_test):
        distinct_id = i

        for strategy in strategies:
            # prepare the request payload with the current strategy
            payload = {
                "bike_id": bike_id,
                "continent_id": continent_id,
                "bike_type": bike_type,
                "category": category,
                "distinct_id": distinct_id,
                "family_id": family_id,
                "price": price,
                "frame_size_code": frame_size_code,
                "rider_height": rider_height,
                "n": n,
                "strategy": strategy,  # set the current strategy
            }
            # simulate a post request to the recommendation endpoint
            start_time = time.time()
            response = client.post("/recommendation", json=payload)
            end_time = time.time()

            # ensure the request was successful
            assert (
                response.status_code == 200
            ), f"request failed with status code {response.status_code} for strategy {strategy}"
            # parse the response data
            data = response.json()
            recommendation = data.get("recommendation")
            strategy_used = data.get("strategy")
            # Check if the strategy is 'homepage' and verify the corresponding strategy
            if strategy == "homepage":
                expected_strategy = strategy_dict[
                    strategy
                ].__name__  # Get the class name as a string
                assert (
                    strategy_used == expected_strategy
                ), f"expected strategy {expected_strategy}, got {strategy_used} for 'product_page'"
            # check the time taken for the recommendation
            assert (
                end_time - start_time < limit
            ), f"{strategy} took {(end_time - start_time)*1000} ms, limit is {limit*1000} ms"
            # assert that the response has the expected length n
            assert (
                len(recommendation) == n
            ), f"expected {n} recommendations for strategy {strategy}, got {len(recommendation)}"
            assert all(isinstance(item, int) for item in recommendation), \
                f"expected all recommendations to be integers for strategy {strategy}, got {recommendation}"
            # ... (other assertions or checks based on the response)


def test_integration_fast_time_len_strats_bikes(inputs_fastapi, limit=50, n_test=100):
    """test time and len of return for all strategies and a random bike_ids of the fastapi app"""
    (
        bike_id,
        continent_id,
        bike_type,
        category,
        distinct_id,
        family_id,
        price,
        frame_size_code,
        rider_height_min,
        rider_height_max,
        rider_height,
        n,
        sample,
        ratio,
        client,
        logger,
        strategy_dict,
    ) = inputs_fastapi
    strategies = list(strategy_dict.keys())

    # exclude braze and homepage since they do not ensure returning n
    to_remove = ["braze", "homepage", "bot"]
    # Remove items from strategies if they exist
    strategies = [item for item in strategies if item not in to_remove]

    random_bike_ids = np.random.choice(50000, size=n_test, replace=False).tolist()

    for bike_id in random_bike_ids:
        for strategy in strategies:
            # prepare the request payload with the current strategy
            payload = {
                "bike_id": bike_id,
                "continent_id": continent_id,
                "bike_type": bike_type,
                "category": category,
                "distinct_id": distinct_id,
                "family_id": family_id,
                "price": price,
                "frame_size_code": frame_size_code,
                "rider_height": rider_height,
                "n": n,
                "strategy": strategy,  # set the current strategy
            }
            # simulate a post request to the recommendation endpoint
            start_time = time.time()
            response = client.post("/recommendation", json=payload)
            end_time = time.time()

            # ensure the request was successful
            assert (
                response.status_code == 200
            ), f"request failed with status code {response.status_code} for strategy {strategy}"
            # parse the response data
            data = response.json()
            recommendation = data.get("recommendation")
            strategy_used = data.get("strategy")
            # Check if the strategy is 'product_page' and verify the corresponding strategy
            if strategy == "product_page":
                expected_strategy = strategy_dict[
                    strategy
                ].__name__  # Get the class name as a string
                assert (
                    strategy_used == expected_strategy
                ), f"expected strategy {expected_strategy}, got {strategy_used} for 'product_page'"
            # check the time taken for the recommendation
            assert (
                end_time - start_time < limit
            ), f"{strategy} took {(end_time - start_time)*1000} ms, limit is {limit*1000} ms"
            # assert that the response has the expected length n
            assert (
                len(recommendation) == n
            ), f"expected {n} recommendations for strategy {strategy}, got {len(recommendation)}"
            assert all(isinstance(item, int) for item in recommendation), \
                f"expected all recommendations to be integers for strategy {strategy}, got {recommendation}"
            # ... (other assertions or checks based on the response)
def test_integration_bot_strategy(inputs_fastapi, limit=100):
    """Test the 'bot' strategy of the fastapi app to ensure it returns a list of size n with string elements."""
    (
        bike_id,
        continent_id,
        bike_type,
        category,
        distinct_id,
        family_id,
        price,
        frame_size_code,
        rider_height_min,
        rider_height_max,
        rider_height,
        n,
        sample,
        ratio,
        client,
        logger,
        strategy_dict,
    ) = inputs_fastapi
    # Prepare the request payload with the 'bot' strategy
    payload = {
        "bike_id": bike_id,
        "continent_id": continent_id,
        "bike_type": bike_type,
        "category": category,
        "distinct_id": distinct_id,
        "family_id": family_id,
        "price": price,
        "frame_size_code": frame_size_code,
        "rider_height": rider_height,
        "n": n,
        "strategy": "bot",  # Set the strategy to 'bot'
    }
    # Simulate a post request to the recommendation endpoint
    start_time = time.time()
    response = client.post("/recommendation", json=payload)
    end_time = time.time()
    # Ensure the request was successful
    assert (
        response.status_code == 200
    ), f"request failed with status code {response.status_code} for strategy 'bot'"
    # Parse the response data
    data = response.json()
    recommendation = data.get("recommendation")
    strategy_used = data.get("strategy")
    # Print the output of the returned model
    print(f"Strategy used: {strategy_used}")
    print(f"Recommendation: {recommendation}")
    # Check the time taken for the recommendation
    assert (
        end_time - start_time < limit
    ), f"'bot' strategy took {(end_time - start_time)*1000} ms, limit is {limit*1000} ms"
    # Assert that the response has the expected length n
    assert (
        len(recommendation) == n
    ), f"expected {n} recommendations for strategy 'bot', got {len(recommendation)}"
    # Assert that all elements in the recommendation list are strings
    assert all(isinstance(item, str) for item in recommendation), \
        f"expected all recommendations to be strings for strategy 'bot', got {recommendation}"
    # ... (other assertions or checks based on the response)

