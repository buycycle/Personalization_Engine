import pandas as pd


import boto3
import gzip

import os
import configparser

import pickle

import datetime

from buycycle.data import aws_client


def s3_credentials(config_paths: str = "config/config.ini"):
    """read the config file and return the s3 credentials"""

    config = configparser.ConfigParser()
    config.read(config_paths)

    Bucket = config["S3"]["bucket"]
    Key = config["S3"]["key"]
    Filename = config["S3"]["file"]

    return Bucket, Key, Filename


def download_s3(client: boto3.client, Bucket: str, Key: str, filename: str, path: str, newFilename: str):
    """download a file from s3 and move it to a folder and rename it
    Args:
        client (boto3.client): s3 client
        Bucket (str): s3 bucket
        Key (str): s3 key
        filename (str): filename to download
        path (str): path to download to
        newFilename (str): new filename of download_file

    """

    Key = Key + filename
    client.download_file(Bucket, Key, filename)

    # move the file to the folder and rename with NewFilename
    os.rename(filename, path + newFilename)


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


def aggreate(df: pd.DataFrame, user_id: str, bike_id: str, user_features: list, item_features: list) -> pd.DataFrame:
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
    implicit_feedback, features, user_features, item_features, user_id, bike_id, file_name="data/export.json.gz", fraction=0.8
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


def get_multi_day_data(
    start_date: datetime,
    end_date: datetime,
    implicit_feedback: dict,
    features: list[str],
    user_features: list[str],
    item_features: list[str],
    user_id: str,
    bike_id: str,
    Bucket: str,
    keysbase: str = "2892805/",
    inclusive: str = "both",
    fraction: float = 0.8,
):
    """
    Generate the multi-day test dataset using data from specified start to end dates.

    Args:
        start_date (datetime): The start date for the test data.
        end_date (datetime): The end date for the test data.
        features (list): The feature list for the dataset.
        user_features (list): The list of user features for the dataset.
        item_features (list): The list of item features for the dataset.
        inclusive (str, optional): Whether to include the start and end dates in the dataset, defaults to 'both'.
        fraction (float, optional): A parameter used somewhere in your function, defaults to 0.3.

    Returns:
        df (pd.DataFrame): Pandas DataFrame containing event and features data.
        metadata (datetime): Latest date of the dataset
    """

    client = aws_client()

    datelist = pd.date_range(start=start_date, end=end_date, freq="D", inclusive=inclusive)
    # generate Keys in the following format '2892805/2023/06/18/full_day/'
    Keys = [keysbase + date.strftime("%Y/%m/%d/full_day/") for date in datelist]

    df = pd.DataFrame()

    for Key in Keys:
        download_s3(client, Bucket, Key, filename="export.json.gz", path="data/", newFilename="export.json.gz")

        df = pd.concat(
            [
                df,
                read_extract_local_data(
                    implicit_feedback, features, user_features, item_features, user_id, bike_id, fraction=fraction
                ),
            ],
            axis=0,
        )
        df = aggreate(df, user_id, bike_id, user_features, item_features)

    metadata = end_date

    return df, metadata


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


def get_last_date_local(path, limit=10):
    """get last date as saved to metadata.pkl, if the metadata.pkl does not exist, return current date minus limit
    Args:
        path (str): path to metadata.pkl
        limit (int): limit in days to subtract from current date
    Returns:
        last_date (datetime): last date as saved to metadata.pkl or current date minus limit
    """
    if os.path.isfile(path + "metadata.pkl"):
        with open(path + "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        return metadata
    else:
        return datetime.date.today() - datetime.timedelta(days=limit)


def get_last_date_S3(client: aws_client, Bucket: str, filename: str = "export.json.gz") -> datetime.date:
    """get last date of data in S3, determined by last modified minus one day
    Args:
        client (boto3.client): boto3 client
        Bucket (str): name of S3 bucket
        filename (str): name of file to look for
    Returns:
        last_date (datetime): last date of data in S3
    """

    response = client.list_objects_v2(Bucket=Bucket)
    if "Contents" in response:
        # Filter objects so it only contains those end with 'export.json.gz'
        contents = [obj for obj in response["Contents"] if obj["Key"].endswith(filename)]
        # Get the max 'LastModified' among the filtered objects
        if contents:
            latest_obj = max(contents, key=lambda x: x["LastModified"])
        else:
            return None
    # subtract one day from the last modified date since the data is updated the following day
    last_date = latest_obj["LastModified"].date() - datetime.timedelta(days=1)

    return last_date
