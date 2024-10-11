"""
Functional tests for the FastAPI model response.
"""
import time
import random
import numpy as np
from src.driver_content import prefilter_features
from src.content import get_mask_continent, get_user_preference_mask
from src.strategies import ContentMixed

# Constants
EXCLUDED_STRATEGIES = ["braze", "bot"]
PRODUCT_PAGE_STRATEGY = ["product_page"]
COLLAB_STRATEGY = ["homepage"]
LIMIT_MS = 100
N_TEST_USERS = 2
N_TEST_BIKES = 100


def create_payload(inputs, strategy, bike_id=None, user_id=None):
    """Create a payload for the recommendation request."""
    return {
        "bike_id": bike_id or inputs["bike_id"],
        "continent_id": inputs["continent_id"],
        "bike_type": inputs["bike_type"],
        "category": inputs["category"],
        "user_id": user_id or inputs["user_id"],
        "family_id": inputs["family_id"],
        "price": inputs["price"],
        "frame_size_code": inputs["frame_size_code"],
        "rider_height": inputs["rider_height"],
        "n": inputs["n"],
        "strategy": strategy,
    }


def post_request(client, payload):
    """Simulate a POST request to the recommendation endpoint and return the response and elapsed time."""
    start_time = time.time()
    response = client.post("/recommendation", json=payload)
    end_time = time.time()
    return response, end_time - start_time


def assert_response(response, strategy, elapsed_time, limit):
    """Assert the response status, time."""
    assert (
        response.status_code == 200
    ), f"Request failed with status code {response.status_code} for strategy {strategy}"
    data = response.json()
    recommendation = data.get("recommendation")
    assert (
        elapsed_time < limit
    ), f"{strategy} took {elapsed_time * 1000} ms, limit is {limit * 1000} ms"


def assert_recommendation_length(recommendation, strategy, expected_length):
    """Assert the length."""
    assert (
        len(recommendation) == expected_length
    ), f"Expected {expected_length} recommendations for strategy {strategy}, got {len(recommendation)}"


def test_integration_fast_time_strats_input(inputs, limit=LIMIT_MS):
    """Test time and length of return for all strategies of the FastAPI app."""
    strategies = [
        s for s in inputs["strategy_dict"].keys() if s not in EXCLUDED_STRATEGIES
    ]
    limit /= 1000  # Convert limit to seconds
    for strategy in strategies:
        payload = create_payload(inputs, strategy)
        response, elapsed_time = post_request(inputs["client"], payload)
        assert_response(response, strategy, elapsed_time, limit)


def test_integration_fast_time_strats_collab_users(
    inputs, testdata_collaborative, limit=LIMIT_MS, n_test=N_TEST_USERS
):
    """Test time and length of return for all strategies and a random subsample of collaborative users."""
    strategies = COLLAB_STRATEGY
    # Filter users to include only those with IDs shorter than 10 characters
    users = [
        user_id
        for user_id in testdata_collaborative.dataset.mapping()[0].keys()
        if len(user_id) < 10
    ]
    for user_id in random.sample(users, n_test):
        for strategy in strategies:
            payload = create_payload(inputs, strategy, user_id=user_id)
            response, elapsed_time = post_request(inputs["client"], payload)
            assert_response(response, strategy, elapsed_time, limit)
            assert_recommendation_length(
                response.json().get("recommendation"), strategy, inputs["n"]
            )


def test_integration_fast_time_len_strats_bikes(
    inputs, limit=LIMIT_MS, n_test=N_TEST_BIKES
):
    """Test time and length of return for all strategies and a random sample of bike_ids."""
    strategies = PRODUCT_PAGE_STRATEGY
    random_bike_ids = np.random.choice(50000, size=n_test, replace=False).tolist()
    for bike_id in random_bike_ids:
        for strategy in strategies:
            payload = create_payload(inputs, strategy, bike_id=bike_id)
            response, elapsed_time = post_request(inputs["client"], payload)
            assert_response(response, strategy, elapsed_time, limit)
            assert_recommendation_length(
                response.json().get("recommendation"), strategy, inputs["n"]
            )


def test_integration_bot_strategy(inputs, limit=LIMIT_MS):
    """Test the 'bot' strategy of the FastAPI app to ensure it returns a list of size n with string elements."""
    payload = create_payload(inputs, "bot")
    response, elapsed_time = post_request(inputs["client"], payload)
    assert (
        response.status_code == 200
    ), f"Request failed with status code {response.status_code} for strategy 'bot'"
    data = response.json()
    recommendation = data.get("recommendation")
    assert (
        elapsed_time < limit / 1000
    ), f"'bot' strategy took {elapsed_time * 1000} ms, limit is {limit} ms"
    assert (
        len(recommendation) == inputs["n"]
    ), f"Expected {inputs['n']} recommendations for strategy 'bot', got {len(recommendation)}"
    assert all(
        isinstance(item, str) for item in recommendation
    ), f"Expected all recommendations to be strings for strategy 'bot', got {recommendation}"


def test_recommendations_fit_preference_mask_with_user_preferences(
    inputs,
    testdata_content,
    testdata_collaborative,
    n_test_users=N_TEST_USERS,
    n_test_bikes=N_TEST_BIKES,
):
    """Test that recommendations fit the preference mask, including user-specific preferences."""
    data_store_content = testdata_content
    strategies = PRODUCT_PAGE_STRATEGY + COLLAB_STRATEGY
    # Filter users to include only those with IDs shorter than 10 characters
    users = [
        user_id
        for user_id in testdata_collaborative.dataset.mapping()[0].keys()
        if len(user_id) < 10
    ]
    # add here to filter out users that we also have preferences for
    sampled_users = random.sample(users, n_test_users)
    # Get a random sample of bike_ids
    random_bike_ids = np.random.choice(50000, size=n_test_bikes, replace=False).tolist()
    for user_id in sampled_users:
        for bike_id in random_bike_ids:
            for strategy in strategies:
                payload = create_payload(
                    inputs, strategy, bike_id=bike_id, user_id=user_id
                )
                # Simulate a POST request to the recommendation endpoint
                response, _ = post_request(inputs["client"], payload)
                # Ensure the request was successful
                assert (
                    response.status_code == 200
                ), f"Request failed with status code {response.status_code}"
                # Get the recommendations from the response
                data = response.json()
                recommendations = data.get("recommendation")
                # Assert the length of the recommendations
                assert_recommendation_length(recommendations, strategy, inputs["n"])
                # Recreate the preference mask logic from the model using testdata_content
                # Get general and user-specific preference masks
                preference_mask = get_mask_continent(
                    data_store_content, inputs["continent_id"]
                )
                preference_mask_user = get_user_preference_mask(
                    data_store_content, user_id, strategy
                )
                # Combine general and user-specific preference masks
                if preference_mask_user:
                    preference_mask_set = set(preference_mask)
                    preference_mask_user_set = set(preference_mask_user)
                    combined_mask = preference_mask_set.intersection(
                        preference_mask_user_set
                    )
                    preference_mask = sorted(list(combined_mask))
                    # Check that all recommendations fit the combined preference mask
                    for recommendation in recommendations:
                        assert (
                            recommendation in preference_mask
                        ), f"Recommendation {recommendation} does not fit the preference mask"
