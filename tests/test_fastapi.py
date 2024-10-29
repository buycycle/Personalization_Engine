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
LIMIT_MS = 200
N_TEST_USERS = 2
N_TEST_BIKES = 10

random.seed(1)

def create_payload(inputs, strategy, bike_id=None, user_id=None):
    """Create a payload for the recommendation request."""
    return {
        "bike_id": bike_id or inputs["bike_id"],
        "user_id": user_id or inputs["user_id"],
        "n": inputs["n"],
        "strategy": strategy,
    }


def post_request(client, payload):
    """Simulate a POST request to the recommendation endpoint and return the response and elapsed time."""
    start_time = time.time()
    response = client.post("/recommendation", json=payload)
    end_time = time.time()
    return response, end_time - start_time


def assert_response(response, payload, elapsed_time, limit):
    """Assert the response status and time."""
    assert response.status_code == 200, (
        f"Request failed with status code {response.status_code}.\n"
        f"Payload: {payload}\n"
        f"Response: {response.text}"
    )

    data = response.json()
    recommendation = data.get("recommendation")

    assert elapsed_time < limit, (
        f"Request exceeded time limit.\n"
        f"Payload: {payload}\n"
        f"Elapsed Time: {elapsed_time * 1000:.2f} ms\n"
        f"Time Limit: {limit * 1000:.2f} ms"
    )


def assert_recommendation_length(recommendation, payload, expected_length):
    """Assert the length of the recommendation list."""
    assert len(recommendation) == expected_length, (
        f"Expected {expected_length} recommendations "
        f"but got {len(recommendation)}.\n"
        f"Payload: {payload}"
    )

def test_integration_fast_time_strats_input(inputs, limit=LIMIT_MS):
    """Test time and length of return for all strategies of the FastAPI app."""
    strategies = [
        s for s in inputs["strategy_dict"].keys() if s not in EXCLUDED_STRATEGIES
    ]
    limit /= 1000  # Convert limit to seconds
    for strategy in strategies:
        payload = create_payload(inputs, strategy)
        response, elapsed_time = post_request(inputs["client"], payload)
        assert_response(response, payload, elapsed_time, limit)


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
            user_id = int(user_id)
            payload = create_payload(inputs, strategy, user_id=user_id)
            response, elapsed_time = post_request(inputs["client"], payload)
            assert_response(response, payload, elapsed_time, limit)

def test_integration_fast_time_len_strats_random_bikes(
    inputs, limit=LIMIT_MS, n_test=N_TEST_BIKES
):
    """Test time and length of return for all strategies and a random sample of bike_ids."""
    strategies = PRODUCT_PAGE_STRATEGY
    random_bike_ids = np.random.choice(50000, size=n_test, replace=False).tolist()
    for bike_id in random_bike_ids:
        for strategy in strategies:
            payload = create_payload(inputs, strategy, bike_id=bike_id)


            response, elapsed_time = post_request(inputs["client"], payload)


            assert_response(response, payload, elapsed_time, limit)
            assert_recommendation_length(
                response.json().get("recommendation"), payload, inputs["n"]
            )



def test_integration_fast_time_len_strats_bikes(
    inputs, testdata_content, limit=LIMIT_MS, n_test_bikes=N_TEST_BIKES
):
    """Test time and length of return for all strategies and a sample of bike_ids."""
    strategies = PRODUCT_PAGE_STRATEGY
    bike_ids = testdata_content.similarity_matrix.cols
    # Ensure we don't exceed the number of available bike IDs
    n_test = min(n_test_bikes, len(bike_ids))

    # Use the first n_test bike IDs from the DataFrame
    test_bike_ids = bike_ids[:n_test]

    for bike_id in test_bike_ids:
        for strategy in strategies:
            payload = create_payload(inputs, strategy, bike_id=int(bike_id))


            response, elapsed_time = post_request(inputs["client"], payload)

            assert_response(response, payload, elapsed_time, limit)
            assert_recommendation_length(
                response.json().get("recommendation"), payload, inputs["n"]
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


def test_recommendations_fit_preference_mask_with_user_preferences_active_bikes(
    inputs,
    testdata_content,
    testdata_collaborative,
    n_test_users=N_TEST_USERS,
    n_test_bikes=N_TEST_BIKES,
    limit=LIMIT_MS,
):
    """
    Test that recommendations for known bikes fit the preference mask, including user-specific preferences.
    Test on the intersection of users we have preferences for and a collaborative filtering strategy.
    """
    data_store_content = testdata_content
    strategies = PRODUCT_PAGE_STRATEGY + COLLAB_STRATEGY
    # Filter users to include are in the collaborative model and have preference
    preference_user_ids = set(data_store_content.df_preference_user.index)
    collaborative_user_ids = set(testdata_collaborative.dataset.mapping()[0].keys())
    # Find the intersection of both sets
    valid_user_ids = collaborative_user_ids.intersection(preference_user_ids)
    valid_user_ids = list(valid_user_ids)
    # Sample users from the intersection
    test_users = random.sample(valid_user_ids, min(n_test_users, len(valid_user_ids)))
    # active bikes
    bike_ids = testdata_content.df_status_masked.index.tolist()
    # Ensure we don't exceed the number of available bike IDs
    n_test = min(n_test_bikes, len(bike_ids))
    # Use the first n_test bike IDs from the DataFrame
    test_bike_ids = bike_ids[:n_test]
    for user_id in test_users:
        for bike_id in test_bike_ids:
            for strategy in strategies:
                user_id = int(user_id)
                payload = create_payload(
                    inputs, strategy, bike_id=bike_id, user_id=user_id
                )
                response, elapsed_time = post_request(inputs["client"], payload)
                assert_response(response, payload, elapsed_time, limit)
                # Get the recommendations from the response
                data = response.json()
                recommendations = data.get("recommendation")
                # Assert the length of the recommendations
                assert_recommendation_length(
                    response.json().get("recommendation"), payload, inputs["n"]
                )
                # Recreate the preference mask logic from the model using testdata_content
                # Get general and user-specific preference masks
                preference_mask = get_mask_continent(
                    data_store_content, inputs["continent_id"])
                preference_mask_user = get_user_preference_mask(
                    data_store_content, user_id, strategy
                )
                preference_mask = set(preference_mask)
                preference_mask_user = set(preference_mask_user)
                # Combine general and user-specific preference masks
                preference_mask = preference_mask.intersection(preference_mask_user)
                for recommendation in recommendations:
                    assert (
                        recommendation in preference_mask
                    ), (
                        f"Recommendation {recommendation} does not fit the preference mask.\n"
                        f"payload: {payload}\n"
                        f"Preference Mask: {preference_mask}\n"
                        f"Recommendations: {recommendations}"
                    )
                    # Assert that the recommendation is an active bike
                    assert (
                        recommendation in bike_ids
                    ), (
                        f"Recommendation {recommendation} is not an active bike.\n"
                        f"Active Bikes: {bike_ids}\n"
                        f"Recommendations: {recommendations}"
                    )

