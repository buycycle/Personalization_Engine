"""test fixutres used in the tests"""

import os
import pytest

from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from fastapi import FastAPI

from buycycle.logger import KafkaLogger
from src.data_content import read_data_content

# for create_data
from src.data_content import create_data_model_content
from src.data_content import DataStoreContent
from src.collaborative import DataStoreCollaborative

from src.driver_content import (
    main_query,
    main_query_dtype,
    popularity_query,
    popularity_query_dtype,
    categorical_features,
    numerical_features,
    prefilter_features,
    numerical_features_to_overweight,
    numerical_features_overweight_factor,
    categorical_features_to_overweight,
    categorical_features_overweight_factor,
)

from src.driver_collaborative import user_id, bike_id, item_features, user_features, query

from src.collaborative import create_data_model_collaborative, update_model, read_data_model
from src.data_collaborative import write_data, read_data_collaborative

from src.strategies import strategy_dict

# get loggers
from buycycle.logger import Logger


@pytest.fixture(scope="package")
def mock_kafka_logger():
    "mock the KafkaLogger"
    # Create a mock KafkaLogger instance
    mock_logger = Mock(spec=KafkaLogger)

    # Configure the mock to behave as you expect
    # For example, you can set return values for methods or assert that methods are called
    mock_logger.info.return_value = None
    mock_logger.error.return_value = None
    mock_logger.warning.return_value = None
    mock_logger.debug.return_value = None
    return mock_logger


@pytest.fixture(scope="package")
def app_mock(mock_kafka_logger):
    "patch the model KafkaLogger with the mock version and prevent threads from starting"

    with patch("buycycle.logger.KafkaLogger", return_value=mock_kafka_logger), patch("kafka.KafkaProducer"), patch(
        "src.data_content.DataStoreContent.read_data_periodically"
    ), patch("src.collaborative.DataStoreCollaborative.read_data_periodically"):
        # The above patches will replace the actual methods with mocks that do nothing
        from model.app import app  # Import inside the patch context to apply the mock

        yield app  # Use yield to make it a fixture


@pytest.fixture(scope="package")
def inputs(app_mock, mock_kafka_logger):
    "inputs for the function unit tests"

    logger = mock_kafka_logger
    app = app_mock

    bike_id = 22187
    distinct_id = "1234"
    family_id = 1101
    price = 1200
    frame_size_code = "56"
    n = 12
    sample = 50
    ratio = 0.5
    # Create a TestClient for your FastAPI app
    client = TestClient(app)

    return bike_id, distinct_id, family_id, price, frame_size_code, n, sample, ratio, client, logger


@pytest.fixture(scope="package")
def inputs_fastapi(app_mock, mock_kafka_logger):
    "inputs for the fastapi function test"

    logger = mock_kafka_logger

    app = app_mock

    strategies = list(strategy_dict.keys())

    bike_id = 14394
    distinct_id = "1234"
    family_id = 1101
    price = 1200
    frame_size_code = "56"
    n = 12
    sample = 50
    ratio = 0.5
    # Create a TestClient for your FastAPI app
    client = TestClient(app)

    return bike_id, distinct_id, family_id, price, frame_size_code, n, sample, ratio, client, logger, strategies


@pytest.fixture(scope="package")
def testdata_content():
    # make folder data if not exists
    if not os.path.exists("./data/"):
        os.makedirs("./data/")

    create_data_model_content(
        main_query + "LIMIT 1000", # limit to subset for integation testing
        main_query_dtype,
        popularity_query,
        popularity_query_dtype,
        categorical_features,
        numerical_features,
        prefilter_features,
        numerical_features_to_overweight,
        numerical_features_overweight_factor,
        categorical_features_to_overweight,
        categorical_features_overweight_factor,
        status=["active"],
        metric="euclidean",
        path="./data/",
    )

    # create data stores
    data_store_content = DataStoreContent(prefilter_features=prefilter_features)

    data_store_content.read_data()

    return data_store_content


@pytest.fixture(scope="package")
def testdata_collaborative():
    # make folder data if not exists
    if not os.path.exists("./data/"):
        os.makedirs("./data/")

    create_data_model_collaborative(
        DB="DB_EVENTS",
        driver="snowflake",
        query=query + "LIMIT 10000", # limit to subset for integation testing
        user_id=user_id,
        bike_id=bike_id,
        user_features=user_features,
        item_features=item_features,
        update_model=update_model,
        path="./data/",
    )

    data_store_collaborative = DataStoreCollaborative()

    data_store_collaborative.read_data()

    return data_store_collaborative
