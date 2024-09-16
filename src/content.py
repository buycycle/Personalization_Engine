import pandas as pd
import random

from typing import Tuple, List, Optional

from src.helper import interveave


def get_top_n_quality_prefiltered_bot(
        df_quality: pd.DataFrame, preference_mask_set: set, filter_features: tuple, n: int = 16
) -> Tuple[List[str], Optional[str]]:
    """
    Returns the top n recommendations based on quality, progressively filtering for price, frame_size_code, and family_id
    Args:
        df_quality (pd.DataFrame): DataFrame with sorted bike ids by quality
        preference_mask_set (set): bike indices matching preferences
        filter_features (tuple): filtering features and filter condition
        n (int): number of recommendations to return
    Returns:
        Tuple[List[str], Optional[str]]: list of top n bike ids by quality and an error message if any
    """
    error = None
    try:
        # Apply preference mask filter
        df_filtered = df_quality[df_quality.index.isin(preference_mask_set)]
        df_filtered = df_filtered.sample(frac=0.5)
        last_valid_df = df_filtered  # Keep track of the last valid DataFrame
        # Apply additional filters progressively
        for feature, condition in filter_features:
            df_temp = df_filtered[condition(df_filtered)]
            # introduce some variance
            if len(df_temp) >= n*2:
                last_valid_df = df_temp  # Update the last valid DataFrame
            else:
                break  # Stop filtering if we have less than n elements
            df_filtered = df_temp  # Apply the current filter
        # Return the top n*2 from the last valid filtered DataFrame
        df_top_n = last_valid_df.head(n*2)
        # to introduce some variance in the results, sample n from the top n*2
        df_sampled = df_top_n.sample(n=n)
        top_n_recommendations = df_sampled.slug.tolist()
        return top_n_recommendations, error
    except Exception as e:
        error = str(e)
        return [], error  # Return an empty list if an exception occurs

def get_top_n_quality_prefiltered(
    df_quality: pd.DataFrame,
    preference_mask: list,
    bike_type: int,
    family_id: int,
    price: int,
    frame_size_code: str,
    n: int = 16,
) -> list:
    """
    Returns the top n recommendations based on quality, progressively filtering for price, frame_size_code, and family_id
    Args:
        df_quality (pd.DataFrame): DataFrame with sorted bike ids by quality
        preference_mask (list): bike indicies matching preferences
        bike_type (int): bike_type of the bike
        family_id (int): family_id of the bike
        price (int): price of the bike
        frame_size_code (str): frame_size_code of the bike
        n (int): number of recommendations to return
    Returns:
        list: list of top n bike ids by quality
    """
    df_quality_preference = df_quality[df_quality.index.isin(preference_mask)]
    # Filter for 20% higher and lower price
    df_filtered_bike_type = df_quality_preference[
        df_quality_preference["bike_type"] == bike_type
    ]
    # Filter for 20% higher and lower price
    df_filtered_price = df_filtered_bike_type[
        (df_filtered_bike_type["price"] >= price * 0.8)
        & (df_filtered_bike_type["price"] <= price * 1.2)
    ]
    # Filter for same frame_size_code
    df_filtered_size = df_filtered_price[
        df_filtered_price["frame_size_code"] == frame_size_code
    ]
    # Filter for same family_id
    df_filtered_family = df_filtered_size[df_filtered_size["family_id"] == family_id]
    # Step-wise approach to get at least n elements
    if len(df_filtered_family) >= n:
        return df_filtered_family.head(n).index.tolist()
    elif len(df_filtered_size) >= n:
        return df_filtered_size.head(n).index.tolist()
    elif len(df_filtered_price) >= n:
        return df_filtered_price.head(n).index.tolist()
    elif len(df_filtered_bike_type) >= n:
        return df_filtered_bike_type.head(n).index.tolist()
    elif len(df_quality_preference) >= n:
        return df_quality_preference.head(n).index.tolist()
    else:
        return df_quality.head(n).index.tolist()


def get_top_n_recommendations(
    bike_similarity_df: pd.DataFrame, bike_id: int, preference_mask: list, n: int = 16
) -> list:
    """
    Returns the top n recommendations for a bike_id, given a bike_similarity_df
    Args:
        bike_similarity_df (pd.DataFrame): bike similarity df
        bike_id (int): bike_id to get recommendations for
        preference_mask (list): preference mask
        n (int): number of recommendations to return

    Returns:
        list: list of top n recommendations for bike_id, skipping the bike_id itself


    """

    bike_similarity_df = bike_similarity_df[
        bike_similarity_df.index.isin(preference_mask)
    ]

    # squeeze to convert pd.DataFrame into pd.Series
    return bike_similarity_df.loc[bike_id].squeeze().nsmallest(n + 1).index.tolist()


def get_top_n_recommendations_prefiltered(
    bike_similarity_df: pd.DataFrame,
    preference_mask: list,
    df: pd.DataFrame,
    df_status_masked: pd.DataFrame,
    bike_id: int,
    prefilter_features: list,
    logger,
    n: int = 16,
) -> list:
    """
    Returns the top n recommendations for a bike_id with a prefilter applied for the prefilter_features of the specific bike_id
    empty list if only the bike_id itself or less match the prefilter
    Args:
        bike_similarity_df (pd.DataFrame): cosine bike similarity df
        preference_mask (list): bike indicies matching preferences
        df (pd.DataFrame): dataframe of bikes
        df_status_masked (pd.DataFrame): dataframe of bikes with the given status
        bike_id (int): bike_id to get recommendations for
        n (int): number of recommendations to return
    Returns:
        list: list of top n recommendations for bike_id, skipping the bike_id itself
    """

    # get the values of the prefilter_features for the bike_id
    df_status_masked = df_status_masked[df_status_masked.index.isin(preference_mask)]

    prefilter_values = df.loc[bike_id, prefilter_features]

    assert len(prefilter_features) == len(
        prefilter_values
    ), "Features and values must have the same length"

    # mask for df_status_masked to include only where all prefilter_features are true
    mask = (df_status_masked[prefilter_features] == prefilter_values).all(axis=1)
    true_ids = mask[mask].index

    # Use the true_ids to filter columns in bike_similarity_df for the given bike_id
    filtered_similarity_df = bike_similarity_df.loc[bike_id, true_ids]

    # Check if the resulting filtered DataFrame has more than one column
    if len(filtered_similarity_df) > 1:
        # If so, return the n smallest values' indices as a list
        return filtered_similarity_df.squeeze().nsmallest(n).index.tolist()
    else:
        # If not, return an empty list
        return []


def get_top_n_recommendations_mix(
    bike_id: int,
    preference_mask: list,
    filter_features: tuple,
    bike_type: int,
    family_id: int,
    price: int,
    frame_size_code: str,
    df: pd.DataFrame,
    df_status_masked: pd.DataFrame,
    df_quality: pd.DataFrame,
    bike_similarity_df: pd.DataFrame,
    prefilter_features: List[str],
    logger,
    n: int = 16,
    sample: int = 100,
    ratio: float = 0.75,
    interveave_prefilter_general: bool = True,
) -> Tuple[List[int], Optional[str]]:
    """
    Mix of quality and content-based recommendations.
    Logic:
        1. Get the top n recommendations based on quality; return if bike_id not in the df.
        2. Get the top n recommendations prefiltered by the prefilter_features for n * ratio.
        3. Get the top n recommendations for the bike_id for n.
        4. Intervene or append the lists in the order of 2, 3, and append 1; ensuring that enough recommendations are returned.
    Args:
        bike_id (int): Bike ID to get recommendations for.
        preference_mask (list): bike indicies matching preferences
        bike_type (int): Bike type used for filtering quality recommendations.
        family_id (int): Family ID used for filtering quality recommendations.
        price (int): Price used for filtering quality recommendations.
        frame_size_code (str): Frame size code used for filtering quality recommendations.
        df (pd.DataFrame): Dataframe of bikes.
        df_status_masked (pd.DataFrame): Dataframe of bikes with the given status.
        df_quality (pd.DataFrame): Dataframe of bikes sorted by quality.
        bike_similarity_df (pd.DataFrame): Bike similarity dataframe.
        prefilter_features (List[str]): List of features to prefilter by.
        logger (logging.Logger): Logger for logging warnings and errors.
        n (int, optional): Number of recommendations to return. Defaults to 16.
        sample (int, optional): Number of top_n_quality recommendations to randomly sample from. Defaults to 100.
        ratio (float, optional): Ratio of prefiltered recommendations to generic recommendations. Defaults to 0.75.
        interveave_prefilter_general (bool, optional): If True, interweave prefiltered and generic recommendations, else append. Defaults to True.
    Returns:
        Tuple[List[int], Optional[str]]: A tuple containing a list of top n recommendations for bike_id and an error message, if any.
    """

    error = None
    top_n_recommendations = []

    try:
        if bike_id not in df.index:

            top_n_quality = get_top_n_quality_prefiltered(
                df_quality,
                preference_mask,
                bike_type,
                family_id,
                price,
                frame_size_code,
                sample,
            )

            return random.sample(top_n_quality, n), error

        else:
            # prefiltered recommendations
            top_n_recommendations_prefiltered = get_top_n_recommendations_prefiltered(
                bike_similarity_df,
                preference_mask,
                df,
                df_status_masked,
                bike_id,
                prefilter_features,
                logger,
                int(n * ratio),
            )

            # get the top n recommendations for the bike_id
            top_n_recommendations_generic = get_top_n_recommendations(
                bike_similarity_df, bike_id, preference_mask, n
            )

            # remove bike_id from recommendations, we do not want to recommend the same bike
            try:
                top_n_recommendations_generic.remove(bike_id)
            except:
                pass
            try:
                top_n_recommendations_prefiltered.remove(bike_id)
            except:
                pass

            # if prefiltered recommendations exist
            if top_n_recommendations_prefiltered:
                # interveave prefiltered and generic recommendations
                if interveave_prefilter_general:
                    top_n_recommendations = interveave(
                        top_n_recommendations_prefiltered, top_n_recommendations_generic
                    )

                else:
                    top_n_recommendations = (
                        top_n_recommendations_prefiltered
                        + top_n_recommendations_generic
                    )

            else:
                top_n_recommendations = top_n_recommendations_generic

            # remove duplicates
            top_n_recommendations = list(dict.fromkeys(top_n_recommendations))

            # return the top n recommendations
            return top_n_recommendations[:n], error

    except Exception as e:
        error = str(e)
        return top_n_recommendations, error
