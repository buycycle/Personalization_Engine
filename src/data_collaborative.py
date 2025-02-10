import pandas as pd


import gzip

import os
import configparser

import pickle

import datetime


def unpack_data(file_name):
    """unpack the data from the .gz file"""

    return gzip.open(file_name, "rb")


def parse_json(file_name):
    return pd.read_json(file_name, lines=True)


def extract_data(df: pd.DataFrame, event: list, features: list) -> pd.DataFrame:
    """
    Extracts event and features data from a pd.DataFrame and returns a pd.DataFrame.
    Filters for events.

    Args:
        df (pd.DataFrame): Pandas DataFrame containing event and features data.
        event (list): List of strings containing the event names to extract.
        features (list): List of strings containing features names to extract.

    Returns:
        Pandas DataFrame, contains event and extracted features data.
    """

    df = df[df["event"].isin(event)].reset_index(drop=True)
    # here we parse the whole massive json to only use 3 columns, think about optimization
    df = pd.concat([df.event, pd.json_normalize(df.properties)[features]], axis=1)

    return df


def map_feedback(df: pd, mapping: dict) -> pd.DataFrame:
    """map the df to a feedback dcitionary

    Args:
        df (pd.DataFrame): Pandas DataFrame containing event and features data.
        mapping (dict): dictionary containing the mapping of the events to the feedback

    Returns:
        Pandas DataFrame, contains event, extracted features data and feedback
    """

    df["feedback"] = df["event"].map(mapping)
    return df


def clean_data(df: pd.DataFrame, user_id: str, interaction_limit: int = 1000) -> pd.DataFrame:
    """clean data

    Args:
        df (pd.DataFrame): Pandas DataFrame containing event and features data.
        user_id (str): name of the user_id column
    Returns:
        Pandas DataFrame, contains event, extracted features data and feedback

    """

    # delete rows of distinct_id which have more than interaction_limit rows
    df["count"] = df.groupby(user_id)[user_id].transform("count")
    df = df[df["count"] <= interaction_limit]

    # this might not be necessary and curenlt does not work
    #  df = frame_size_code_to_numeric(
    #      df, bike_type_id_column=bike_id, frame_size_code_column="bike.frame_size")

    return df


def aggreate(
    df: pd.DataFrame,
    user_id: str,
    bike_id: str,
    user_features: list,
    item_features: list,
) -> pd.DataFrame:
    features = user_features + item_features

    df = df.copy()

    # convert all item features to string
    for feature in features:
        df[feature] = df[feature].astype("str")

    agg_function = {"feedback": "sum"}
    for feature in features:
        agg_function[feature] = pd.Series.mode

    df = df.groupby([user_id, bike_id]).agg(agg_function).reset_index()

    return df


def read_extract_local_data(
    implicit_feedback,
    features,
    user_features,
    item_features,
    user_id,
    bike_id,
    file_name="data/export.json.gz",
    fraction=0.8,
):
    """
    read the data from the local folder, extract the relevant features, clean and aggregate
    Args:
        implicit_feedback (dict): dictionary containing the mapping of the events to the feedback
        file_name (str): name of the file
        features (list): list of features to extract
        user_features (list): list of user features to extract
        item_features (list): list of item features to extract
        bike_id (str): name of the bike_id column
        fraction (float): fraction of the data to use
    Returns:
        Pandas DataFrame, contains event, extracted features data and feedback
    """

    # upack the json file
    json_file = unpack_data(file_name)
    df = parse_json(json_file)
    df = df.sample(frac=fraction, random_state=1)

    # extract relevant data
    event = list(implicit_feedback.keys())
    df = extract_data(df, event, features)
    df.dropna(inplace=True)
    df = map_feedback(df, implicit_feedback)

    df = clean_data(df, user_id)

    df = aggreate(df, user_id, bike_id, user_features, item_features)

    return df


def combine_data(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    user_id: str,
    bike_id: str,
    user_features: list[str],
    item_features: list[str],
) -> pd.DataFrame:
    """combine two dataframes containing event and features data
    Args:
        df1 (pd.DataFrame): Pandas DataFrame containing event and features data.
        df2 (pd.DataFrame): Pandas DataFrame containing event and features data.
        features (list): list of features to extract from the json
        user_features (list): list of user features
        item_features (list): list of item features
    Returns:
        Pandas DataFrame, contains event, extracted features data and feedback
    """

    df = pd.concat([df1, df2], axis=0)
    df = aggreate(df, user_id, bike_id, user_features, item_features)

    return df


def write_data(df, metadata, path):
    df.to_pickle(path + "df_collaborative.pkl")
    # write the metadata str to metadata.pkl
    with open(path + "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)


def read_data_collaborative(path):
    df = pd.read_pickle(path + "df_collaborative.pkl")
    with open(path + "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return df, metadata
