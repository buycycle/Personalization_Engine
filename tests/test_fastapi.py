"""
Functional tests for the FastAPI model response.
"""
import time
import random
import numpy as np
from src.driver_content import prefilter_features
from src.strategies import ContentMixed
from collections import namedtuple

# Constants
EXCLUDED_STRATEGIES = ["braze", "homepage", "bot"]
LIMIT_MS = 100
N_TEST_USERS = 10
N_TEST_BIKES = 100


# Define a named tuple for the inputs
InputsFastAPI = namedtuple(
    "InputsFastAPI",
    [
        "bike_id",
        "continent_id",
        "bike_type",
        "category",
        "distinct_id",
        "family_id",
        "price",
        "frame_size_code",
        "rider_height_min",
        "rider_height_max",
        "rider_height",
        "n",
        "sample",
        "ratio",
        "client",
        "logger",
        "strategy_dict",
    ],
)

def create_payload(inputs, strategy, bike_id=None, distinct_id=None):
    """Create a payload for the recommendation request."""
    return {
        "bike_id": bike_id or inputs.bike_id,
        "continent_id": inputs.continent_id,
        "bike_type": inputs.bike_type,
        "category": inputs.category,
        "distinct_id": distinct_id or inputs.distinct_id,
        "family_id": inputs.family_id,
        "price": inputs.price,
        "frame_size_code": inputs.frame_size_code,
        "rider_height": inputs.rider_height,
        "n": inputs.n,
        "strategy": strategy,
    }

def post_request(client, payload):
    """Simulate a POST request to the recommendation endpoint and return the response and elapsed time."""
    start_time = time.time()
    response = client.post("/recommendation", json=payload)
    end_time = time.time()
    return response, end_time - start_time


def assert_response(response, strategy, n, elapsed_time, limit):
    """Assert the response status, time, and content."""
    assert (
        response.status_code == 200
    ), f"Request failed with status code {response.status_code} for strategy {strategy}"
    data = response.json()
    recommendation = data.get("recommendation")
    assert (
        elapsed_time < limit
    ), f"{strategy} took {elapsed_time * 1000} ms, limit is {limit * 1000} ms"
    assert (
        len(recommendation) == n
    ), f"Expected {n} recommendations for strategy {strategy}, got {len(recommendation)}"
    assert all(
        isinstance(item, int) for item in recommendation
    ), f"Expected all recommendations to be integers for strategy {strategy}, got {recommendation}"


def get_inputs_fastapi(inputs_fastapi):
    """Convert inputs_fastapi to a named tuple for easier access."""
    return InputsFastAPI(*inputs_fastapi)


def test_integration_fast_time_len_strats_input(inputs_fastapi, limit=LIMIT_MS):
    """Test time and length of return for all strategies of the FastAPI app."""
    inputs = get_inputs_fastapi(inputs_fastapi)
    strategies = [
        s for s in inputs.strategy_dict.keys() if s not in EXCLUDED_STRATEGIES
    ]
    limit /= 1000  # Convert limit to seconds
    for strategy in strategies:
        payload = create_payload(inputs, strategy)
        response, elapsed_time = post_request(inputs.client, payload)
        assert_response(response, strategy, inputs.n, elapsed_time, limit)


def test_integration_fast_time_len_strats_collab_users(
    inputs_fastapi, testdata_collaborative, limit=50, n_test=N_TEST_USERS
):
    """Test time and length of return for all strategies and a random subsample of collaborative users."""
    inputs = get_inputs_fastapi(inputs_fastapi)
    strategies = [
        s for s in inputs.strategy_dict.keys() if s not in EXCLUDED_STRATEGIES
    ]
    users = list(testdata_collaborative.dataset.mapping()[0].keys())
    for distinct_id in random.sample(users, n_test):
        for strategy in strategies:
            payload = create_payload(inputs, strategy, distinct_id=distinct_id)
            response, elapsed_time = post_request(inputs.client, payload)
            assert_response(response, strategy, inputs.n, elapsed_time, limit)


def test_integration_fast_time_len_strats_bikes(
    inputs_fastapi, limit=50, n_test=N_TEST_BIKES
):
    """Test time and length of return for all strategies and a random sample of bike_ids."""
    inputs = get_inputs_fastapi(inputs_fastapi)
    strategies = [
        s for s in inputs.strategy_dict.keys() if s not in EXCLUDED_STRATEGIES
    ]
    random_bike_ids = np.random.choice(50000, size=n_test, replace=False).tolist()
    for bike_id in random_bike_ids:
        for strategy in strategies:
            payload = create_payload(inputs, strategy, bike_id=bike_id)
            response, elapsed_time = post_request(inputs.client, payload)
            assert_response(response, strategy, inputs.n, elapsed_time, limit)


def test_integration_bot_strategy(inputs_fastapi, limit=LIMIT_MS):
    """Test the 'bot' strategy of the FastAPI app to ensure it returns a list of size n with string elements."""
    inputs = get_inputs_fastapi(inputs_fastapi)
    payload = create_payload(inputs, "bot")
    response, elapsed_time = post_request(inputs.client, payload)
    assert (
        response.status_code == 200
    ), f"Request failed with status code {response.status_code} for strategy 'bot'"
    data = response.json()
    recommendation = data.get("recommendation")
    assert (
        elapsed_time < limit / 1000
    ), f"'bot' strategy took {elapsed_time * 1000} ms, limit is {limit} ms"
    assert (
        len(recommendation) == inputs.n
    ), f"Expected {inputs.n} recommendations for strategy 'bot', got {len(recommendation)}"
    assert all(
        isinstance(item, str) for item in recommendation
    ), f"Expected all recommendations to be strings for strategy 'bot', got {recommendation}"


