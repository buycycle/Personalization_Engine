import pandas as pd
import numpy as np


# similarity estimation
from sklearn.metrics import pairwise_distances_chunked
from scipy.spatial.distance import cdist  # , pdist, squareform


from buycycle.data import sql_db_read, snowflake_sql_db_read, get_data_status_mask,\
    feature_engineering, frame_size_code_to_numeric, DataStoreBase


def get_similarity_matrix_memory(df: pd.DataFrame, df_feature_engineered: pd.DataFrame, metric: str, working_memory: int = 4001) -> pd.DataFrame:
    """
    get the similarity matrix for the dataframe
    only chunck size optimal, with 4gb not good results, we might need to randomly shuffle similarity_matrix matrix to avoid clustering in chunks
    Args:
        df (pandas.DataFrame): feature engineered dataframe of bikes to get cosine similarity matrix for
        df_feature_engineered (pandas.DataFrame): feature engineered dataframe of bikes to get cosine similarity matrix for
        mertric (str): similarity metric
        working_memory (int)
    Returns:
        similarity_matrix (pd.DataFrame): similarity matrix for the dataframe
    """
    # calculate pairwise distances using pairwise_distances_chunked
    n_samples = df_feature_engineered.shape[0]
    similarity_matrix = np.zeros((n_samples, n_samples))

    for chunk in pairwise_distances_chunked(df_feature_engineered, df_feature_engineered, metric=metric, working_memory=working_memory):
        for i, row in enumerate(chunk):
            similarity_matrix[i, :] = row

    # convert the pairwise distances to a similarity matrix
    similarity_matrix = pd.DataFrame(
        similarity_matrix, columns=df.index, index=df.index)

    return similarity_matrix


def get_similarity_matrix_cdist(df: pd.DataFrame, df_feature_engineered: pd.DataFrame, metric: str, status_mask: pd.DataFrame) -> pd.DataFrame:
    """
    get the similarity matrix for the dataframe
    cdist, since we only need to recommend available bikes (status_mask) but need to recommend for all bikes
    Args:
        df (pandas.DataFrame): feature engineered dataframe of bikes to get cosine similarity matrix for
        df_feature_engineered (pandas.DataFrame): feature engineered dataframe of bikes to get cosine similarity matrix for
        metric (str): metric to use for pairwise distances
        status_mask (pandas.DataFrame): mask for the status of the bikes
    Returns:
        similarity_matrix (pandas.DataFrame): similarity matrix

    """
    # get the cosine similarity matrix
    similarity_matrix = pd.DataFrame(
        cdist(df_feature_engineered, df_feature_engineered.loc[status_mask]),
        columns=status_mask,
        index=df.index,
    )

    similarity_matrix = similarity_matrix.astype("float32")

    return similarity_matrix


def get_data(
    main_query: str,
    main_query_dtype: str,
    popularity_query: str,
    popularity_query_dtype: str,
    index_col: str = "id",
    config_paths: str = "config/config.ini",
) -> pd.DataFrame:
    """
    Get main and popularity data, dropna and fillna for motor column
    Args:
        main_query: query to get main data
        popularity_query: query to get popularity data
        config_paths: path to config file
    Returns:
        df: main data
        df_popularity: popularity data
    """

    df = sql_db_read(query=main_query, DB="DB_BIKES", config_paths=config_paths,
                     dtype=main_query_dtype, index_col=index_col)

    df.motor.fillna(df.motor.median(), inplace=True)
    df.dropna(inplace=True)

    df_popularity = snowflake_sql_db_read(
        query=popularity_query, DB="DB_EVENTS", config_paths=config_paths, dtype=popularity_query_dtype, index_col=index_col)
    df_popularity = frame_size_code_to_numeric(df_popularity)
    df_popularity.dropna(inplace=True)

    return df, df_popularity


def create_data_model_content(
    main_query,
    main_query_dtype,
    popularity_query,
    popularity_query_dtype,
    categorical_features,
    numerical_features,
    prefilter_features,
    numerical_features_to_overweight: list,
    numerical_features_overweight_factor: float,
    categorical_features_to_overweight: list,
    categorical_features_overweight_factor: float,
    status: list,
    metric: str = "eucledian",
    path: str = "data/",

):
    """
    Create data and save it to disk
    includes getting the data,
    replacing the frame_size_code with a numeric value,
    feature engineering the data,
    and getting the similarity matrix
    Args:
        path: path to save data

    """

    df, df_popularity = get_data(
        main_query, main_query_dtype, popularity_query, popularity_query_dtype)

    df = frame_size_code_to_numeric(df)

    df_feature_engineered = feature_engineering(df,
                                                categorical_features,
                                                categorical_features_to_overweight,
                                                categorical_features_overweight_factor,
                                                numerical_features,
                                                numerical_features_to_overweight,
                                                numerical_features_overweight_factor)

    status_mask = get_data_status_mask(df, status)

    similarity_matrix = get_similarity_matrix_cdist(
        df, df_feature_engineered, metric, status_mask)

    # reduce the column dimensionality of the similarity matrix by filtering with the status mask
    # similarity_matrix = similarity_matrix[status_mask]

    df_status_masked = df.loc[status_mask]
    df = df[prefilter_features]
    df_status_masked = df_status_masked[prefilter_features]

    # write df, df_popularity, similarity_matrix to disk

    df.to_pickle(path + "df.pkl")  # where to save it, usually as a .pkl
    df_status_masked.to_pickle(path + "df_status_masked.pkl")
    df_popularity.to_pickle(path + "df_popularity.pkl")
    similarity_matrix.to_pickle(
        path + "similarity_matrix.pkl", compression="tar")


def read_data_content(path: str = "data/"):
    """
    Read data from disk
    Args:
        path: path to save data

    Returns:
        df: main data
        df_status_masked: main data with status mask applied
        df_popularity: popularity data
        similarity_matrix: similarity matrix
    """

    df = pd.read_pickle(path + "df.pkl")
    df_status_masked = pd.read_pickle(path + "df_status_masked.pkl")
    df_popularity = pd.read_pickle(path + "df_popularity.pkl")

    similarity_matrix = pd.read_pickle(
        path + "similarity_matrix.pkl", compression="tar")

    return df, df_status_masked, df_popularity,  similarity_matrix


class DataStoreContent(DataStoreBase):
    def __init__(self, prefilter_features):
        super().__init__()
        self.df = None
        self.df_status_masked = None
        self.df_popularity = None
        self.similarity_matrix = None
        self.prefilter_features =  prefilter_features

    def read_data(self):
        self.df, self.df_status_masked, self.df_popularity, self.similarity_matrix = read_data_content()

    def get_logging_info(self):
        return {
            "df_shape": self.df.shape,
            "df_status_masked_shape": self.df_status_masked.shape,
            "df_popularity_shape": self.df_popularity.shape,
            "similarity_matrix_shape": self.similarity_matrix.shape,
        }
