<<<<<<< HEAD
"""
    Test data realted functions, queries the DB for test data
    Unittest class for testing the DB read-in
    Rest in Pytest
"""

import time

import pytest
import unittest

from tests.test_fixtures import inputs, testdata_content

from src.driver_content import main_query, main_query_dtype, popularity_query, popularity_query_dtype, categorical_features, numerical_features, prefilter_features

from buycycle.data import sql_db_read


from src.data_content import get_data


class TestData(unittest.TestCase):

    """unittest used to check how long DB read takes"""
    def setUp(self):
        self.main_query = main_query
        self.main_query_dtype = main_query_dtype
        self.popularity_query = popularity_query
        self.popularity_query_dtype = popularity_query_dtype
        self.config_paths = "config/config.ini"
        self.prefilter_features = prefilter_features
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.df, self.df_popularity = get_data(
            main_query=self.main_query,
            main_query_dtype=self.main_query_dtype,
            popularity_query=self.popularity_query,
            popularity_query_dtype=self.popularity_query_dtype,
            config_paths=self.config_paths,
        )

    def test_get_data_time(self):
        start_time = time.time()
        get_data(
            main_query=self.main_query,
            main_query_dtype=self.main_query_dtype,
            popularity_query=self.popularity_query,
            popularity_query_dtype=self.popularity_query_dtype,
            config_paths=self.config_paths,
        )
        end_time = time.time()
        self.assertLess(end_time - start_time, 120, "get_data() took more than 2 minutes to execute")

def test_get_data_length(testdata_content):

    data_store_content = testdata_content

    assert len(data_store_content.df) >= 10000, f"df has {len(data_store_content.df)} rows, which is less than 10000 rows"
    assert len(data_store_content.df_popularity) >= 7000, f"df_popularity has {len(data_store_content.df_popularity)} rows, which is less than 7000 rows"


def test_columns_in_get_data(testdata_content):
    """ All features are in the DB read-in"""

    data_store_content = testdata_content

    features = prefilter_features + categorical_features + numerical_features

    for feature in prefilter_features:
        assert feature in data_store_content.df.columns, f"{feature} is not in the dataframe"

def test_get_data_na_drop_ratio(testdata_content, config_paths="config/config.ini", index_col="id"):

    data_store_content = testdata_content

    df_all = sql_db_read(query=main_query, DB="DB_BIKES", config_paths=config_paths,
                     dtype=main_query_dtype, index_col=index_col)



    na_ratio = 1 - (len(data_store_content.df) / len(df_all))

    assert na_ratio < 0.1, f"NA ratio is {na_ratio:.2f}, which is above 0.1"

||||||| empty tree
=======
"""
    Test data realted functions, queries the DB for test data
    Unittest class for testing the DB read-in
    Rest in Pytest
"""

import time

import pytest
import unittest


from src.driver_content import (
    main_query,
    main_query_dtype,
    popularity_query,
    popularity_query_dtype,
    categorical_features,
    numerical_features,
    prefilter_features,
)

from buycycle.data import sql_db_read


from src.data_content import get_data


class TestData(unittest.TestCase):

    """unittest used to check how long DB read takes"""

    def setUp(self):
        self.main_query = main_query
        self.main_query_dtype = main_query_dtype
        self.popularity_query = popularity_query
        self.popularity_query_dtype = popularity_query_dtype
        self.config_paths = "config/config.ini"
        self.prefilter_features = prefilter_features
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.df, self.df_popularity = get_data(
            main_query=self.main_query,
            main_query_dtype=self.main_query_dtype,
            popularity_query=self.popularity_query,
            popularity_query_dtype=self.popularity_query_dtype,
            config_paths=self.config_paths,
        )

    def test_get_data_time(self):
        start_time = time.time()
        get_data(
            main_query=self.main_query,
            main_query_dtype=self.main_query_dtype,
            popularity_query=self.popularity_query,
            popularity_query_dtype=self.popularity_query_dtype,
            config_paths=self.config_paths,
        )
        end_time = time.time()
        self.assertLess(end_time - start_time, 120, "get_data() took more than 2 minutes to execute")


def test_get_data_length(testdata_content):
    data_store_content = testdata_content

    assert len(data_store_content.df) >= 10000, f"df has {len(data_store_content.df)} rows, which is less than 10000 rows"
    assert (
        len(data_store_content.df_popularity) >= 7000
    ), f"df_popularity has {len(data_store_content.df_popularity)} rows, which is less than 7000 rows"


def test_columns_in_get_data(testdata_content):
    """All features are in the DB read-in"""

    data_store_content = testdata_content

    features = prefilter_features + categorical_features + numerical_features

    for feature in prefilter_features:
        assert feature in data_store_content.df.columns, f"{feature} is not in the dataframe"


def test_get_data_na_drop_ratio(testdata_content, config_paths="config/config.ini", index_col="id"):
    data_store_content = testdata_content

    df_all = sql_db_read(query=main_query, DB="DB_BIKES", config_paths=config_paths, dtype=main_query_dtype, index_col=index_col)

    na_ratio = 1 - (len(data_store_content.df) / len(df_all))

    assert na_ratio < 0.1, f"NA ratio is {na_ratio:.2f}, which is above 0.1"
>>>>>>> dev
