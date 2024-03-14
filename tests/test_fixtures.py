"""test fixutres used in the tests"""

import os
import pytest

from flask import Flask
from flask.logging import create_logger


from src.data_content import read_data_content

# for create_data
from src.data_content import create_data_model_content
from src.data_content import DataStoreContent
from src.collaborative import DataStoreCollaborative

from src.driver_content import (
    main_query,
    main_query_dtype,
    quality_query,
    quality_query_dtype,
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


@pytest.fixture(scope="package")
def inputs():
    bike_id = 14394
    distinct_id = "1234"
    family_id = 1101
    price = 1200
    frame_size_code = "56"
    n = 12
    sample = 50
    ratio = 0.5
    app = Flask(__name__)
    logger = create_logger(app)

    return bike_id, distinct_id, family_id, price, frame_size_code, n, sample, ratio, app, logger


@pytest.fixture(scope="package")
def testdata_content():
    # make folder data if not exists
    if not os.path.exists("./data/"):
        os.makedirs("./data/")

    create_data_model_content(
        main_query,
        main_query_dtype,
        quality_query,
        quality_query_dtype,
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
        query=query,
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
