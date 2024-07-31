import pandas as pd
import numpy as np
import threading

# similarity estimation
import heapq
from scipy.spatial.distance import cdist  # , pdist, squareform
from scipy.sparse import lil_matrix

from scipy.sparse import csr_matrix


import pickle

from buycycle.data import (
    sql_db_read,
    snowflake_sql_db_read,
    get_data_status_mask,
    feature_engineering,
    frame_size_code_to_numeric,
    DataStoreBase,
)


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


def get_similarity_matrix_cdist_queue(
    df_feature_engineered: pd.DataFrame,
    metric: str,
    status_mask: pd.Series,
    percentile: int = 10,
) -> SimilarityMatrixSparse:
    """
    Get the similarity matrix for the dataframe, only keeping the smallest distances based on the percentile threshold.
    Uses a priority queue to maintain the smallest distances, memory efficient.
    Args:
        df_feature_engineered (pd.DataFrame): The feature engineered dataframe of items to compute the similarity matrix for.
        metric (str): The metric to use for pairwise distances.
        status_mask (pd.Series): A mask for the status of the items, indicating which items are available.
        percentile (int): The percentile of the smallest values to keep in each row of the similarity matrix.
    Returns:
        SimilarityMatrixSparse: A data class containing the sparse similarity matrix, row indices, and column indices.
    """
    df_feature_engineered_status_masked = df_feature_engineered.loc[status_mask]
    # Determine the number of columns to keep based on the percentile
    num_cols_to_keep = max(int(len(status_mask) * (percentile / 100)), 1)
    # Initialize a sparse matrix in 'lil' format for efficient row operations
    similarity_matrix_sparse = lil_matrix(
        (df_feature_engineered.shape[0], df_feature_engineered_status_masked.shape[0]),
        dtype="float32",
    )
    # Create a mapping from the filtered DataFrame's indices to the column indices of the sparse matrix
    col_index_mapping = {
        idx: col_idx
        for col_idx, idx in enumerate(df_feature_engineered_status_masked.index)
    }
    # Compute distances for each row to the df_feature_engineered[status_mask] and use a priority queue to keep only the smallest values
    for i, row in enumerate(df_feature_engineered.itertuples(index=False)):
        distances = cdist([row], df_feature_engineered_status_masked, metric=metric)[0]
        smallest_distances = heapq.nsmallest(
            num_cols_to_keep, enumerate(distances), key=lambda x: x[1]
        )
        # Update the sparse matrix with the smallest distances
        for col_idx, dist in smallest_distances:
            # Map the col_idx to the correct column index in the sparse matrix
            mapped_col_idx = col_index_mapping[
                df_feature_engineered_status_masked.index[col_idx]
            ]
            similarity_matrix_sparse[
                i, mapped_col_idx
            ] = dist  # Use `i` as the row index
    # Convert the 'lil' matrix to 'csr' format after all insertions are done
    similarity_matrix_sparse = similarity_matrix_sparse.tocsr()
    # Return an instance of SimilarityMatrixSparse with the sparse matrix and indices
    return SimilarityMatrixSparse(
        matrix=similarity_matrix_sparse,
        rows=df_feature_engineered.index,  # Original indices are maintained
        cols=df_feature_engineered_status_masked.index,  # Indices of available items
    )


def get_similarity_matrix_cdist(
    df_feature_engineered: pd.DataFrame,
    metric: str,
    status_mask: pd.Series,
    percentile: int = 10,
) -> SimilarityMatrixSparse:
    """
    Get the similarity matrix for the dataframe, only keeping the smallest distances based on the percentile threshold.
    Args:
        df_feature_engineered (pd.DataFrame): The feature engineered dataframe of items to compute the similarity matrix for.
        metric (str): The metric to use for pairwise distances.
        status_mask (pd.Series): A mask for the status of the items, indicating which items are available.
        percentile (int): The percentile of the smallest values to keep in each row of the similarity matrix.
    Returns:
        SimilarityMatrixSparse: A data class containing the sparse similarity matrix, row indices, and column indices.
    """
    df_feature_engineered_status_masked = df_feature_engineered.loc[status_mask]
    num_rows = df_feature_engineered.shape[0]
    num_filtered_cols = df_feature_engineered_status_masked.shape[0]
    # Determine the number of columns to keep based on the percentile
    num_cols_to_keep = max(int(len(status_mask) * (percentile / 100)), 1)
    # Create a mapping from the filtered DataFrame's indices to the column indices of the sparse matrix
    col_index_mapping = {
        idx: col_idx
        for col_idx, idx in enumerate(df_feature_engineered_status_masked.index)
    }
    # Initialize lists to hold the data for the sparse matrix
    similarity_matrix_data = []
    similarity_matrix_rows = []
    similarity_matrix_cols = []
    # Compute distances for each row to the df_feature_engineered[status_mask]
    for i, row in enumerate(df_feature_engineered.values):
        distances = cdist(
            [row], df_feature_engineered_status_masked.values, metric=metric
        )[0]
        # Get the indices of the smallest distances
        smallest_indices = np.argpartition(distances, num_cols_to_keep)[
            :num_cols_to_keep
        ]
        # Get the corresponding smallest distances
        smallest_distances = distances[smallest_indices]
        # Update the lists for the sparse matrix with the smallest distances
        similarity_matrix_data.extend(smallest_distances)
        similarity_matrix_rows.extend([i] * num_cols_to_keep)
        # Map the column indices to the correct column index in the sparse matrix
        similarity_matrix_cols.extend(
            [
                col_index_mapping[df_feature_engineered_status_masked.index[j]]
                for j in smallest_indices
            ]
        )
    # Create the sparse matrix from the lists
    similarity_matrix_sparse = csr_matrix(
        (similarity_matrix_data, (similarity_matrix_rows, similarity_matrix_cols)),
        shape=(num_rows, num_filtered_cols),
        dtype="float32",
    )
    # Return an instance of SimilarityMatrixSparse with the sparse matrix and indices
    return SimilarityMatrixSparse(
        matrix=similarity_matrix_sparse,
        rows=df_feature_engineered.index,  # Original indices are maintained
        cols=df_feature_engineered_status_masked.index,  # Indices of available items
    )


def construct_dense_similarity_row(
    similarity_data: SimilarityMatrixSparse, bike_id: int
) -> pd.DataFrame:
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
        bike_similarity_df = pd.DataFrame(
            bike_dense_vector, index=[bike_id], columns=similarity_data.cols
        )

        return bike_similarity_df, error

    except Exception as e:
        error = str(e)
        return bike_similarity_df, error


def get_data(
    main_query: str,
    main_query_dtype: str,
    quality_query: str,
    quality_query_dtype: str,
    index_col: str = "id",
    config_paths: str = "config/config.ini",
) -> pd.DataFrame:
    """
    Get main and quality data, dropna and fillna for motor column
    Args:
        main_query: query to get main data
        quality_query: query to get quality data
        config_paths: path to config file
    Returns:
        df: main data
        df_quality: quality data
    """

    df = sql_db_read(
        query=main_query,
        DB="DB_BIKES",
        config_paths=config_paths,
        dtype=main_query_dtype,
        index_col=index_col,
    )
    duplicates = df.index.duplicated(keep="last")
    df = df[~duplicates]

    df.motor.fillna(df.motor.median(), inplace=True)
    df.dropna(inplace=True)

    df_quality = sql_db_read(
        query=quality_query,
        DB="DB_BIKES",
        config_paths=config_paths,
        dtype=quality_query_dtype,
        index_col=index_col,
    )

    duplicates = df_quality.index.duplicated(keep="last")
    df_quality = df_quality[~duplicates]

    df_quality = frame_size_code_to_numeric(df_quality, bike_type_id_column="bike_type")
    df_quality.dropna(inplace=True)
    df_quality["rider_height_min"] = (
        df_quality["rider_height_min"].fillna(150).astype("int64")
    )
    df_quality["rider_height_max"] = (
        df_quality["rider_height_max"].fillna(195).astype("int64")
    )

    return df, df_quality


def create_data_model_content(
    main_query,
    main_query_dtype,
    quality_query,
    quality_query_dtype,
    categorical_features,
    numerical_features,
    preference_features,
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

    df, df_quality = get_data(
        main_query, main_query_dtype, quality_query, quality_query_dtype
    )

    df = frame_size_code_to_numeric(df, bike_type_id_column="bike_type")

    df_preference = df[preference_features]

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

    similarity_matrix = get_similarity_matrix_cdist(
        df_feature_engineered, metric, status_mask
    )

    df_status_masked = df.loc[status_mask]
    df = df[prefilter_features]
    df_status_masked = df_status_masked[prefilter_features]

    # write df, df_quality, similarity_matrix to disk

    df.to_pickle(path + "df.pkl")
    df_preference.to_pickle(path + "df_preference.pkl")
    df_status_masked.to_pickle(path + "df_status_masked.pkl")
    df_quality.to_pickle(path + "df_quality.pkl")
    similarity_matrix.to_pickle(path + "similarity_matrix.pkl")


def read_data_content(path: str = "data/"):
    """
    Read data from disk
    Args:
        path: path to save data

    Returns:
        df: main data
        df_preference: df with preference_feature values
        df_status_masked: main data with status mask applied
        df_quality: quality data
        similarity_matrix: similarity matrix
    """

    df = pd.read_pickle(path + "df.pkl")
    df_preference = pd.read_pickle(path + "df_preference.pkl")
    df_status_masked = pd.read_pickle(path + "df_status_masked.pkl")
    df_quality = pd.read_pickle(path + "df_quality.pkl")

    similarity_matrix = SimilarityMatrixSparse.read_pickle(
        path + "similarity_matrix.pkl"
    )

    return df, df_preference, df_status_masked, df_quality, similarity_matrix


class DataStoreContent(DataStoreBase):
    def __init__(self, prefilter_features):
        super().__init__()
        self.df = None
        self.df_preference = None
        self.df_status_masked = None
        self.df_quality = None
        self.similarity_matrix = None
        self.prefilter_features = prefilter_features
        self._lock = threading.Lock()

    def read_data(self):
        with self._lock:  # acquire lock
            (
                self.df,
                self.df_preference,
                self.df_status_masked,
                self.df_quality,
                self.similarity_matrix,
            ) = read_data_content()

    def get_logging_info(self):
        return {
            "df_shape": self.df.shape,
            "df_preference": self.df_preference.shape,
            "df_status_masked_shape": self.df_status_masked.shape,
            "df_quality_shape": self.df_quality.shape,
        }
