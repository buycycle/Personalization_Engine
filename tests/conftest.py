import os
import subprocess
from unittest.mock import Mock, patch
import pytest
from fastapi.testclient import TestClient
from src.data_content import (
    read_data_content,
    create_data_model_content,
    DataStoreContent,
)
from src.collaborative import (
    DataStoreCollaborative,
    create_data_model_collaborative,
    update_model,
)
from src.driver_content import (
    main_query,
    main_query_dtype,
    quality_query,
    quality_query_dtype,
    categorical_features,
    numerical_features,
    preference_features,
    prefilter_features,
    numerical_features_to_overweight,
    numerical_features_overweight_factor,
    categorical_features_to_overweight,
    categorical_features_overweight_factor,
)
from src.driver_collaborative import (
    user_id,
    bike_id,
    item_features,
    user_features,
    query,
)
from src.strategies import strategy_dict
from buycycle.logger import Logger

DATA_PATH = "./data/"
BIKE_ID = 18894
CONTINENT_ID = 1
BIKE_TYPE = 1
CATEGORY = "road"
DISTINCT_ID = "1234"
FAMILY_ID = 2502
PRICE = 1200
FRAME_SIZE_CODE = "56"
RIDER_HEIGHT_MIN = 140
RIDER_HEIGHT_MAX = 195
RIDER_HEIGHT = 180
N = 12
SAMPLE = 50
RATIO = 0.5


@pytest.fixture(scope="package")
def mock_logger():
    """Mock the KafkaLogger."""
    return Mock(spec=Logger)


@pytest.fixture(scope="package")
def app_mock(mock_logger):
    """Patch the model with the logger mock version and prevent threads from starting."""
    with patch("buycycle.logger.Logger", return_value=mock_logger), patch(
        "src.data_content.DataStoreContent.read_data_periodically"
    ), patch("src.collaborative.DataStoreCollaborative.read_data_periodically"):
        from model.app import app

        yield app


@pytest.fixture(scope="package")
def inputs(app_mock, mock_logger):
    """Inputs for the function unit tests."""
    client = TestClient(app_mock)
    return (
        BIKE_ID,
        CONTINENT_ID,
        BIKE_TYPE,
        CATEGORY,
        DISTINCT_ID,
        FAMILY_ID,
        PRICE,
        FRAME_SIZE_CODE,
        RIDER_HEIGHT_MIN,
        RIDER_HEIGHT_MAX,
        RIDER_HEIGHT,
        N,
        SAMPLE,
        RATIO,
        client,
        mock_logger,
    )


@pytest.fixture(scope="package")
def inputs_fastapi(app_mock, mock_logger):
    """Inputs for the FastAPI function test."""
    subprocess.run(["python", "create_data.py", DATA_PATH, "test"], check=True)
    client = TestClient(app_mock)
    return (
        14394,
        CONTINENT_ID,
        BIKE_TYPE,
        CATEGORY,
        DISTINCT_ID,
        1101,
        2000,
        FRAME_SIZE_CODE,
        RIDER_HEIGHT_MIN,
        RIDER_HEIGHT_MAX,
        RIDER_HEIGHT,
        5,
        10,
        RATIO,
        client,
        mock_logger,
        strategy_dict,
    )


@pytest.fixture(scope="package")
def testdata_content():
    """Create and return a DataStoreContent instance for testing."""
    os.makedirs(DATA_PATH, exist_ok=True)
    create_data_model_content(
        main_query + "LIMIT 500",
        main_query_dtype,
        quality_query,
        quality_query_dtype,
        categorical_features,
        numerical_features,
        preference_features,
        prefilter_features,
        numerical_features_to_overweight,
        numerical_features_overweight_factor,
        categorical_features_to_overweight,
        categorical_features_overweight_factor,
        status=["active"],
        metric="euclidean",
        path=DATA_PATH,
    )
    data_store_content = DataStoreContent(prefilter_features=prefilter_features)
    data_store_content.read_data()
    return data_store_content


@pytest.fixture(scope="package")
def testdata_collaborative():
    """Create and return a DataStoreCollaborative instance for testing."""
    os.makedirs(DATA_PATH, exist_ok=True)
    create_data_model_collaborative(
        DB="DB_EVENTS",
        driver="snowflake",
        query=query + "LIMIT 1000",
        user_id=user_id,
        bike_id=bike_id,
        user_features=user_features,
        item_features=item_features,
        update_model=update_model,
        path=DATA_PATH,
    )
    data_store_collaborative = DataStoreCollaborative()
    data_store_collaborative.read_data()
    return data_store_collaborative
