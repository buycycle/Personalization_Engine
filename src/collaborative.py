import numpy as np
import time
import random

from typing import Tuple, List, Optional

from lightfm.evaluation import auc_score
from lightfm.evaluation import precision_at_k
from lightfm.cross_validation import random_train_test_split
from lightfm.data import Dataset
from lightfm import LightFM


import pickle
import threading

from buycycle.data import snowflake_sql_db_read, DataStoreBase


def construct_dataset(df, user_id, bike_id, user_features, item_features):
    """construct the dataset including user and item features"""

    dataset = Dataset()

    features = user_features + item_features

    # convert all features to string
    for feature in features:
        df[feature] = df[feature].astype("str")

    # fit users, items
    dataset.fit(users=df[user_id].unique(), items=df[bike_id].unique())

    # flatten and convert to list all the user features from the dataframe
    user_features_flat = df[user_features].values.ravel("K").tolist()

    # add user features mapping to existing mappings
    dataset.fit_partial(users=df[user_id].unique(), user_features=user_features_flat)

    # flatten and convert to list all the item features from the dataframe
    item_features_flat = df[item_features].values.ravel("K").tolist()

    # add item features mapping to existing mappings
    dataset.fit_partial(items=df[bike_id].unique(), item_features=item_features_flat)

    return dataset


def construct_interactions(df, dataset, user_id, bike_id, feedback):
    """construct the interactions from the iterable [user_id, bike_id, feedback]

    Args:
        df (pd.DataFrame): dataframe
        dataset (lightfm.Dataset): lightfm dataset
        user_id (str): user id column name
        bike_id (str): bike id column name
        feedback (str): feedback column name
    Returns:
        interactions (lightfm.SparseMatrix): lightfm interactions
        interactions_weights (lightfm.SparseMatrix): lightfm interactions weights
    """

    interactions, interactions_weights = dataset.build_interactions(
        df[[user_id, bike_id, feedback]].values
    )

    return interactions, interactions_weights


def construct_item_features(df, dataset, bike_id, item_features):
    """construct the item features from the dataframe

    Args:
        df (pd.DataFrame): dataframe
        dataset (lightfm.Dataset): lightfm dataset
        bike_id (str): bike id column name
        item_features (list): list of item features column names
    Returns:
        item_features_matrix (lightfm.SparseMatrix): lightfm item features matrix

    """

    item_features_tuple_list = []

    for index, row in df.iterrows():
        feature_list = []
        for col in item_features:
            feature_value = row[col]
            feature_list.append(feature_value)
        item_features_tuple_list.append((row[bike_id], feature_list))

    item_features_matrix = dataset.build_item_features(item_features_tuple_list)

    return item_features_matrix


def construct_user_features(df, dataset, bike_id, user_features):
    """construct the user features from the dataframe

    Args:
        df (pd.DataFrame): dataframe
        dataset (lightfm.Dataset): lightfm dataset
        bike_id (str): bike id column name
        user_features (list): list of user features column names
    Returns:
        user_features_matrix (lightfm.SparseMatrix): lightfm user features matrix

    """

    user_features_tuple_list = []

    for index, row in df.iterrows():
        feature_list = []
        for col in user_features:
            feature_value = row[col]
            feature_list.append(feature_value)
        user_features_tuple_list.append((row[bike_id], feature_list))

    user_features_matrix = dataset.build_user_features(user_features_tuple_list)

    return user_features_matrix


def construct_train_test(interactions, interactions_weights, test_percentage=0.05):
    train, test = random_train_test_split(
        interactions,
        test_percentage=test_percentage,
        random_state=np.random.RandomState(3),
    )

    train_weights, test_weights = random_train_test_split(
        interactions_weights,
        test_percentage=test_percentage,
        random_state=np.random.RandomState(3),
    )

    return train, train_weights, test, test_weights


def construct_model(
    train,
    user_features_matrix,
    item_features_matrix,
    weights,
    epochs,
    num_components,
    learning_rate,
    loss,
    random_state=1,
):
    """Initialize a LightFM model instance and fit to the training data
    Args:
        train (lightfm.SparseMatrix): lightfm training interactions
        user_features_matrix (lightfm.SparseMatrix): lightfm user features matrix
        item_features_matrix (lightfm.SparseMatrix): lightfm item features matrix
    Returns:
        model (lightfm.LightFM): lightfm model instance
    """

    model = LightFM(
        learning_rate=learning_rate,
        loss=loss,
        no_components=num_components,
        random_state=random_state,
    )

    model.fit(
        train,
        user_features=user_features_matrix,
        item_features=item_features_matrix,
        sample_weight=weights,
        epochs=epochs,
        num_threads=4,
    )

    return model


def get_model(
    df,
    user_id,
    bike_id,
    user_features,
    item_features,
    feedback="feedback",
    k=100,
    test_percentage=0.05,
    epochs=10,
    num_components=30,
    learning_rate=0.05,
    loss="bpr",
    random_state=1,
):
    """construct necessary datasets and fit model

    Args:
        df (pd.DataFrame): dataframe
        user_id (str): user id column name
        bike_id (str): bike id column name
        user_features (list): list of user features column names
        item_features (list): list of item features column names
    Returns:
        model (lightfm.LightFM): lightfm model instance
        train (lightfm.SparseMatrix): lightfm training interactions
        test (lightfm.SparseMatrix): lightfm test interactions
        dataset (lightfm.Dataset): lightfm dataset
        interactions (lightfm.SparseMatrix): lightfm interactions
        interactions_weights (lightfm.SparseMatrix): lightfm interactions weights
        item_features_matrix (lightfm.SparseMatrix): lightfm item features matrix

    """

    # xxxxx get the flow of data right and check where adjustments are necessary
    # make the bike_id column an integer
    df[bike_id] = df[bike_id].astype(int)

    dataset = construct_dataset(df, user_id, bike_id, user_features, item_features)

    interactions, interactions_weights = construct_interactions(
        df, dataset, user_id, bike_id, feedback
    )

    train, train_weights, test, test_weights = construct_train_test(
        interactions, interactions_weights, test_percentage
    )

    user_features_matrix = construct_user_features(df, dataset, user_id, user_features)

    item_features_matrix = construct_item_features(df, dataset, bike_id, item_features)

    model = construct_model(
        train,
        user_features_matrix,
        item_features_matrix,
        train_weights,
        epochs,
        num_components,
        learning_rate,
        loss,
        random_state,
    )

    precision = precision_at_k(model=model, test_interactions=test, train_interactions=train, k=k, user_features=user_features_matrix, item_features=item_features_matrix, num_threads=4).mean()

    return (
        model,
        train,
        test,
        dataset,
        interactions,
        interactions_weights,
        user_features_matrix,
        item_features_matrix,
        precision,
    )


def update_model(df, user_id, bike_id, user_features, item_features, path):
    """retrain and write model to disk"""
    (
        model,
        train,
        test,
        dataset,
        interactions,
        interactions_weights,
        user_features_matrix,
        item_features_matrix,
        precision_at_k,
    ) = get_model(df, user_id, bike_id, user_features, item_features)

    write_model_data(model, dataset, path)

    return precision_at_k


def auc(model, train, test, user_features_matrix, item_features_matrix, num_threads=4):
    """calculate auc score for train and test data"""

    test_auc = auc_score(
        model,
        test,
        user_features=user_features_matrix,
        item_features=item_features_matrix,
        num_threads=num_threads,
    ).mean()

    return test_auc


def eval_model(
    model, train, test, user_features_matrix, item_features_matrix, k=4, num_threads=4
):
    """calculate precision, train auc and test auc for model"""
    precision = precision_at_k(
        model=model,
        test_interactions=test,
        train_interactions=train,
        item_features=item_features_matrix,
        user_features=user_features_matrix,
        k=k,
        num_threads=num_threads,
        check_intersections=False,
    ).mean()

    test_auc = auc(
        model,
        train,
        test,
        user_features_matrix=user_features_matrix,
        item_features_matrix=item_features_matrix,
        num_threads=num_threads,
    )

    return precision, test_auc


def write_model_data(model, dataset, path):
    with open(path + "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(path + "dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)


def read_data_model(path="data/"):
    with open(path + "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(path + "dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    return model, dataset

def get_top_n_collaborative_randomized(
    model,
    user_id: str,
    preference_mask: set,
    n: int,
    sample: int,
    dataset,
    df_status_masked_set: set,
    logger,
) -> Tuple[List, Optional[str]]:
    try:
        user_mapping, _, item_mapping, _ = dataset.mapping()

        if user_id not in user_mapping:
            return [], None

        # map user_id to user_id in dataset
        user_id_index = user_mapping[user_id]
        n_items = dataset.interactions_shape()[1]
        item_ids = np.arange(n_items)
        scores = model.predict(user_id_index, item_ids)
        top_items = np.argsort(-scores)
        # Map internal item index back to external item ids
        item_index_id_map = {v: c for c, v in item_mapping.items()}
        # Combine masks and filter
        combined_mask = df_status_masked_set & preference_mask
        filtered_top_item_ids = [
            item_index_id_map[item_id]
            for item_id in top_items
            if item_index_id_map[item_id] in combined_mask
        ]
        # If not enough items, fallback to individual masks
        if len(filtered_top_item_ids) < n:
            filtered_top_item_ids = [
                item_index_id_map[item_id]
                for item_id in top_items
                if item_index_id_map[item_id] in df_status_masked_set
            ]
            if len(filtered_top_item_ids) < n:
                filtered_top_item_ids = [
                    item_index_id_map[item_id]
                    for item_id in top_items
                    if item_index_id_map[item_id] in preference_mask
                ]
        # Randomly sample from the top_item_ids to introduce some variance
        top_item_ids = filtered_top_item_ids[:sample]
        random.shuffle(top_item_ids)
        top_n_item_ids = top_item_ids[:n]
        return top_n_item_ids, None
    except Exception as e:
        logger.error(f"Error in get_top_n_collaborative_randomized: {str(e)}")
        return [], str(e)

def get_top_n_collaborative_randomized(
    model,
    user_id: str,
    preference_mask: set,
    n: int,
    sample: int,
    dataset,
    df_status_masked_set: set,
    logger,
) -> Tuple[List, Optional[str]]:
    """
    Retrieve the top k item ids for a given user_id by using model.predict()
    Randomized from a sample of top_item_ids
    If no collaborative results are available return an empty list

    Args:
        model (LightFM): Trained LightFM model.
        user_id (str): user_id for which to retrieve top k items.
        preference_mask(set): bike indicies matching preferences
        n (int): Number of top items to retrieve.
        sample (int): number of samples to randomize on
        dataset (Dataset): LightFM dataset object containing mapping between internal and external ids.
        df_status_masked_set (set): status masked dataframe
        logger (Logger): Logger object.

    Returns:
        list: List of top n item ids for the given user.
        str: Error message if any.
    """
    error = None
    top_n_item_ids = []
    try:
        if user_id not in dataset.mapping()[0]:
            return (
                top_n_item_ids,
                error,
            )
        # map user_id to user_id in dataset
        user_id_index = dataset.mapping()[0][user_id]

        n_items = dataset.interactions_shape()[1]

        item_ids = np.arange(n_items)

        scores = model.predict(user_id_index, item_ids)

        top_items = np.argsort(-scores)

        # Map internal item index back to external item ids
        item_index_id_map = {v: c for c, v in dataset.mapping()[2].items()}

        # apply status mask if it resturns enough items
        filtered_top_item_ids = [
            item_index_id_map[item_id]
            for item_id in top_items
            if item_index_id_map[item_id] in df_status_masked_set
        ]
        if len(filtered_top_item_ids) > n:
            top_item_ids = filtered_top_item_ids

        # apply preference mask if it resturns enough items
        filtered_top_item_ids = [
            item_index_id_map[item_id]
            for item_id in top_items
            if item_index_id_map[item_id] in preference_mask
        ]
        if len(filtered_top_item_ids) > n:
            top_item_ids = filtered_top_item_ids

        # randomly sample from the top_item_ids to introduce some variance
        top_item_ids = top_item_ids[:sample]
        random.shuffle(top_item_ids)
        top_n_item_ids = top_item_ids[:n]

        return top_n_item_ids, error

    except Exception as e:
        error = str(e)
        return top_n_item_ids, error


def create_data_model_collaborative(
    DB,
    driver,
    query,
    user_id,
    bike_id,
    user_features,
    item_features,
    update_model,
    path: str = "data/",
):
    """create data and train  model for collaborative filtering, write to disc
    Args:
        DB (str): name of the database
        driver (str): name of the driver
        query (str): query to read data from database
        user_id (str): name of the column containing the user id
        bike_id (str): name of the column containing the bike id
        user_features (list): list of user features
        item_features (list): list of item features
        update_model (function): function to update the model
        path (str): path to save the model
    """

    df = snowflake_sql_db_read(query=query, DB=DB, driver=driver, index_col=user_id)

    df = df.dropna()
    df = df.reset_index()

    precision_at_k = update_model(df, user_id, bike_id, user_features, item_features, path)

    return precision_at_k


class DataStoreCollaborative(DataStoreBase):
    def __init__(self):
        super().__init__()
        self.model = None
        self.dataset = None
        self._lock = threading.Lock()

    def read_data(self):
        with self._lock:  # acquire lock
            self.model, self.dataset = read_data_model()

    def get_logging_info(self):
        return {"model_info": str(self.model), "dataset_info": str(self.dataset)}
