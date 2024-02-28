import pandas as pd
import numpy as np

import threading

# similarity estimation
import heapq
from sklearn.metrics import pairwise_distances_chunked
from scipy.spatial.distance import cdist  # , pdist, squareform
from scipy.sparse import lil_matrix

from scipy.sparse import csr_matrix

from dataclasses import dataclass

import pickle

from buycycle.data import (
    sql_db_read,
    snowflake_sql_db_read,
    get_data_status_mask,
    feature_engineering,
    frame_size_code_to_numeric,
    DataStoreBase,
)


def get_similarity_matrix_memory(
    df: pd.DataFrame, df_feature_engineered: pd.DataFrame, metric: str, working_memory: int = 4001
) -> pd.DataFrame:
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

    for chunk in pairwise_distances_chunked(
        df_feature_engineered, df_feature_engineered, metric=metric, working_memory=working_memory
    ):
        for i, row in enumerate(chunk):
            similarity_matrix[i, :] = row

    # convert the pairwise distances to a similarity matrix
    similarity_matrix = pd.DataFrame(similarity_matrix, columns=df.index, index=df.index)

    return similarity_matrix


class SimilarityMatrixSparse:
    """
    A data class to store a sparse similarity matrix along with its row and column indices.
    Attributes:
        matrix (csr_matrix): The sparse similarity matrix.
        rows (pd.Index): The row indices corresponding to the original data.
        cols (pd.Index): The column indices corresponding to the filtered data based on some criteria (e.g., available items).
    """

    def __init__(self, matrix: csr_matrix, rows: pd.Index, cols: pd.Index):
        self.matrix = matrix
        self.rows = rows
        self.cols = cols

    def to_pickle(self, file_path: str):
        """
        Serializes the object to a pickle file.
        Args:
            file_path (str): The path to the file where the object should be saved.
        """
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def read_pickle(file_path: str):
        """
        Deserializes the object from a pickle file.
        Args:
            file_path (str): The path to the pickle file to read.
        Returns:
            SimilarityMatrixSparse: The deserialized SimilarityMatrixSparse object.
        """
        with open(file_path, "rb") as f:
            return pickle.load(f)

def get_similarity_matrix_cdist(
    df_feature_engineered: pd.DataFrame, metric: str, status_mask: pd.Series, percentile: int = 10
) -> SimilarityMatrixSparse:
    """
    Get the similarity matrix for the dataframe, only keeping the lowest percentile for each row.
    Uses the cdist function to compute distances only for available items (status_mask) but for all items in the dataframe.

    Args:
        df (pd.DataFrame): The original dataframe of items.
        df_feature_engineered (pd.DataFrame): The feature engineered dataframe of items to compute the similarity matrix for.
        metric (str): The metric to use for pairwise distances. If empty, Euclidean distance is used.
        status_mask (pd.Series): A mask for the status of the items, indicating which items are available.
        percentile (int): The percentile of the smallest values to keep in each row of the similarity matrix.

    Returns:
        SimilarityMatrixSparse: A data class containing the sparse similarity matrix, row indices, and column indices.
    """
    # Compute the distance matrix using the specified metric or default to 'euclidean' if not provided
    similarity_matrix = pd.DataFrame(
        cdist(df_feature_engineered, df_feature_engineered.loc[status_mask]),
        index=df_feature_engineered.index,
        columns=df_feature_engineered.loc[status_mask].index,
    )
    similarity_matrix = similarity_matrix.astype("float32")
    # Calculate the threshold for the smallest values for each row based on the given percentile
    thresholds = similarity_matrix.apply(lambda row: np.percentile(row, percentile), axis=1)
    # Apply the threshold to each row, setting values above the threshold to 0
    for i, threshold in enumerate(thresholds):
        similarity_matrix.iloc[i, similarity_matrix.iloc[i, :] > threshold] = 0
    # Convert the DataFrame to a sparse matrix
    similarity_matrix_sparse = csr_matrix(similarity_matrix.values)

    # return an instance of SimilarityMatrixSparse with the sparse matrix and indices
    return SimilarityMatrixSparse(matrix=similarity_matrix_sparse, rows=similarity_matrix.index, cols=similarity_matrix.columns)




def get_similarity_matrix_cdist_queue(
    df_feature_engineered: pd.DataFrame, metric: str, status_mask: pd.Series, percentile: int = 10
) -> SimilarityMatrixSparse:
    """
    Get the similarity matrix for the dataframe, only keeping the smallest distances based on the percentile threshold.
    Uses a priority queue to maintain the smallest distances.
    Args:
        df_feature_engineered (pd.DataFrame): The feature engineered dataframe of items to compute the similarity matrix for.
        metric (str): The metric to use for pairwise distances.
        status_mask (pd.Series): A mask for the status of the items, indicating which items are available.
        percentile (int): The percentile of the smallest values to keep in each row of the similarity matrix.
    Returns:
        SimilarityMatrixSparse: A data class containing the sparse similarity matrix, row indices, and column indices.
    """
    # Determine the number of columns to keep based on the percentile
    num_cols_to_keep = max(int(np.sum(status_mask) * (percentile / 100)), 1)
    # Initialize a sparse matrix in 'lil' format for efficient row operations
    similarity_matrix_sparse = lil_matrix((df_feature_engineered.shape[0], df_feature_engineered.loc[status_mask].shape[0]), dtype='float32')
    # Create a mapping from the filtered DataFrame's indices to the column indices of the sparse matrix
    col_index_mapping = {idx: col_idx for col_idx, idx in enumerate(df_feature_engineered.loc[status_mask].index)}
    # Compute distances and use a priority queue to keep only the smallest values
    for i, row in enumerate(df_feature_engineered.itertuples(index=False)):
        distances = cdist([row], df_feature_engineered.loc[status_mask], metric=metric)[0]
        smallest_distances = heapq.nsmallest(num_cols_to_keep, enumerate(distances), key=lambda x: x[1])
        # Update the sparse matrix with the smallest distances
        for col_idx, dist in smallest_distances:
            # Map the col_idx to the correct column index in the sparse matrix
            mapped_col_idx = col_index_mapping[df_feature_engineered.loc[status_mask].index[col_idx]]
            similarity_matrix_sparse[i, mapped_col_idx] = dist  # Use `i` as the row index
    # Convert the 'lil' matrix to 'csr' format after all insertions are done
    similarity_matrix_sparse = similarity_matrix_sparse.tocsr()
    # Return an instance of SimilarityMatrixSparse with the sparse matrix and indices
    return SimilarityMatrixSparse(
        matrix=similarity_matrix_sparse,
        rows=df_feature_engineered.index,  # Original indices are maintained
        cols=df_feature_engineered.loc[status_mask].index  # Indices of available items
    )

def construct_dense_similarity_row(similarity_data: SimilarityMatrixSparse, bike_id: int) -> pd.DataFrame:
    """
    Reconstruct the dense similarity matrix row for a specific bike_id from a sparse similarity matrix.

    Args:
        similarity_data (SimilarityMatrixSparse): The SimilarityMatrixSparse data class instance containing the sparse similarity matrix and indices.
        bike_id (int): The index of the bike for whom to reconstruct the similarity matrix row.

    Returns:
        bike_similarity_df (pd.DataFrame): A DataFrame containing the reconstructed dense similarity row for the specified bike_id.
        error

    """
    bike_similarity_df = None
    error = None

    try:
        bike_position = similarity_data.rows.get_loc(bike_id)
        # Extract the sparse row vector for the specified bike
        bike_sparse_vector = similarity_data.matrix.getrow(bike_position)

        # Convert the sparse row vector to a dense format (numpy array)
        bike_dense_vector = bike_sparse_vector.toarray()

        # Replace all 0s with 1s in the dense vector
        bike_dense_vector[bike_dense_vector == 0] = np.inf

        bike_position = similarity_data.rows.get_loc(bike_id)
        # Extract the sparse row vector for the specified bike
        bike_sparse_vector = similarity_data.matrix.getrow(bike_position)

        # Convert the sparse row vector to a dense format (numpy array)
        bike_dense_vector = bike_sparse_vector.toarray()

        # Replace all 0s with 1s in the dense vector
        bike_dense_vector[bike_dense_vector == 0] = np.inf

        # Create a DataFrame for the single row, using the column indices
        bike_similarity_df = pd.DataFrame(bike_dense_vector, index=[bike_id], columns=similarity_data.cols)

        return bike_similarity_df, error

    except Exception as e:
        error = str(e)
        return bike_similarity_df, error

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

    df = sql_db_read(query=main_query, DB="DB_BIKES", config_paths=config_paths, dtype=main_query_dtype, index_col=index_col)
    duplicates = df.index.duplicated(keep='last')
    df = df[~duplicates]

    df.motor.fillna(df.motor.median(), inplace=True)
    df.dropna(inplace=True)

    df_popularity = snowflake_sql_db_read(
        query=popularity_query, DB="DB_EVENTS", config_paths=config_paths, dtype=popularity_query_dtype, index_col=index_col
    )

    duplicates = df_popularity.index.duplicated(keep='last')
    df_popularity = df_popularity[~duplicates]

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

    df, df_popularity = get_data(main_query, main_query_dtype, popularity_query, popularity_query_dtype)

    df = frame_size_code_to_numeric(df)

    df_feature_engineered = feature_engineering(
        df,
        categorical_features,
        categorical_features_to_overweight,
        categorical_features_overweight_factor,
        numerical_features,
        numerical_features_to_overweight,
        numerical_features_overweight_factor,
    )

    status_mask = get_data_status_mask(df, status)

    similarity_matrix = get_similarity_matrix_cdist_queue(df_feature_engineered, metric, status_mask)

    # reduce the column dimensionality of the similarity matrix by filtering with the status mask
    # similarity_matrix = similarity_matrix[status_mask]

    df_status_masked = df.loc[status_mask]
    df = df[prefilter_features]
    df_status_masked = df_status_masked[prefilter_features]

    # write df, df_popularity, similarity_matrix to disk

    df.to_pickle(path + "df.pkl")  # where to save it, usually as a .pkl
    df_status_masked.to_pickle(path + "df_status_masked.pkl")
    df_popularity.to_pickle(path + "df_popularity.pkl")
    similarity_matrix.to_pickle(path + "similarity_matrix.pkl")


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

    similarity_matrix = SimilarityMatrixSparse.read_pickle(path + "similarity_matrix.pkl")

    return df, df_status_masked, df_popularity, similarity_matrix


class DataStoreContent(DataStoreBase):
    def __init__(self, prefilter_features):
        super().__init__()
        self.df = None
        self.df_status_masked = None
        self.df_popularity = None
        self.similarity_matrix = None
        self.prefilter_features = prefilter_features
        self._lock = threading.Lock()

    def read_data(self):
        with self._lock:  # acquire lock
            self.df, self.df_status_masked, self.df_popularity, self.similarity_matrix = read_data_content()

    def get_logging_info(self):
        return {
            "df_shape": self.df.shape,
            "df_status_masked_shape": self.df_status_masked.shape,
            "df_popularity_shape": self.df_popularity.shape,
        }
