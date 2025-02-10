import pandas as pd
import random

from typing import Tuple, List, Optional

from src.helper import interveave

from buycycle.data import (
    get_numeric_frame_size,
    get_preference_mask,
    get_preference_mask_condition,
    get_preference_mask_condition_list,
)


def get_top_n_quality_prefiltered_bot(
    df_quality: pd.DataFrame,
    preference_mask: set,
    filter_features: tuple,
    n: int = 16,
) -> Tuple[List[str], Optional[str]]:
    """
    Returns the top n recommendations based on quality, progressively filtering for price, frame_size_code, and family_id
    Args:
        df_quality (pd.DataFrame): DataFrame with sorted bike ids by quality
        preference_mask(set): bike indices matching preferences
        filter_features (tuple): filtering features and filter condition
        n (int): number of recommendations to return
    Returns:
        Tuple[List[str], Optional[str]]: list of top n bike ids by quality and an error message if any
    """
    error = None
    try:
        # Check if the filtered DataFrame has at least n rows
        filtered_df = df_quality[df_quality.index.isin(preference_mask)]
        if len(filtered_df) >= n:
            df_filtered = filtered_df.sample(frac=0.5)
        else:
            df_filtered = df_quality

        last_valid_df = df_filtered  # Keep track of the last valid DataFrame
        # Apply additional filters progressively
        for feature, condition in filter_features:
            df_temp = df_filtered[condition(df_filtered)]
            # introduce some variance
            if len(df_temp) >= n * 2:
                last_valid_df = df_temp  # Update the last valid DataFrame
            else:
                break  # Stop filtering if we have less than n elements
            df_filtered = df_temp  # Apply the current filter
        # Return the top n*2 from the last valid filtered DataFrame
        df_top_n = last_valid_df.head(n * 2)
        # to introduce some variance in the results, sample n from the top n*2
        df_sampled = df_top_n.sample(n=n)
        top_n_recommendations = df_sampled.slug.tolist()
        return top_n_recommendations, error
    except Exception as e:
        error = str(e)
        return [], error  # Return an empty list if an exception occurs

def get_top_n_quality_prefiltered_index(
    df_quality: pd.DataFrame,
    preference_mask: set,
    filter_features: tuple,
    n: int = 16,
) -> Tuple[List[str], Optional[str]]:
    """
    Returns the top n recommendations based on quality, progressively filtering for price, frame_size_code, and family_id
    Args:
        df_quality (pd.DataFrame): DataFrame with sorted bike ids by quality
        preference_mask(set): bike indices matching preferences
        filter_features (tuple): filtering features and filter condition
        n (int): number of recommendations to return
    Returns:
        Tuple[List[str], Optional[str]]: list of top n bike ids by quality and an error message if any
    """
    error = None
    try:
        # Check if the filtered DataFrame has at least n rows
        filtered_df = df_quality[df_quality.index.isin(preference_mask)]
        if len(filtered_df) >= n:
            df_filtered = filtered_df.sample(frac=0.5)
        else:
            df_filtered = df_quality

        last_valid_df = df_filtered  # Keep track of the last valid DataFrame
        # Apply additional filters progressively
        for feature, condition in filter_features:
            df_temp = df_filtered[condition(df_filtered)]
            # introduce some variance
            if len(df_temp) >= n * 2:
                last_valid_df = df_temp  # Update the last valid DataFrame
            else:
                break  # Stop filtering if we have less than n elements
            df_filtered = df_temp  # Apply the current filter
        # Return the top n*2 from the last valid filtered DataFrame
        df_top_n = last_valid_df.head(n * 2)
        # to introduce some variance in the results, sample n from the top n*2
        df_sampled = df_top_n.sample(n=n)
        top_n_recommendations = df_sampled.index.tolist()
        return top_n_recommendations, error
    except Exception as e:
        error = str(e)
        return [], error  # Return an empty list if an exception occurs


def get_top_n_quality_prefiltered(
    df_quality: pd.DataFrame,
    preference_mask: set,
    bike_type: int,
    family_id: int,
    price: int,
    frame_size_code: str,
    n: int = 16,
) -> list:
    """
    Returns the top n recommendations based on quality, progressively filtering for price, frame_size_code, and family_id.
    Args:
        df_quality (pd.DataFrame): DataFrame with sorted bike ids by quality.
        preference_mask (set): Bike indices matching preferences.
        bike_type (int): Bike type of the bike.
        family_id (int): Family ID of the bike.
        price (int): Price of the bike.
        frame_size_code (str): Frame size code of the bike.
        n (int): Number of recommendations to return.
    Returns:
        list: List of top n bike ids by quality.
    """
    # Apply all filters at once
    mask = (
        df_quality.index.isin(preference_mask)
        & (df_quality["bike_type"] == bike_type)
        & (df_quality["price"] >= price * 0.8)
        & (df_quality["price"] <= price * 1.2)
        & (df_quality["frame_size_code"] == frame_size_code)
        & (df_quality["family_id"] == family_id)
    )
    df_filtered = df_quality[mask]
    # Step-wise approach to get at least n elements
    if len(df_filtered) >= n:
        return df_filtered.head(n).index.tolist()
    # Relax filters progressively
    mask_size = (
        df_quality.index.isin(preference_mask)
        & (df_quality["bike_type"] == bike_type)
        & (df_quality["price"] >= price * 0.8)
        & (df_quality["price"] <= price * 1.2)
        & (df_quality["frame_size_code"] == frame_size_code)
    )
    df_filtered_size = df_quality[mask_size]
    if len(df_filtered_size) >= n:
        return df_filtered_size.head(n).index.tolist()
    mask_price = (
        df_quality.index.isin(preference_mask)
        & (df_quality["bike_type"] == bike_type)
        & (df_quality["price"] >= price * 0.8)
        & (df_quality["price"] <= price * 1.2)
    )
    df_filtered_price = df_quality[mask_price]
    if len(df_filtered_price) >= n:
        return df_filtered_price.head(n).index.tolist()
    mask_bike_type = df_quality.index.isin(preference_mask) & (df_quality["bike_type"] == bike_type)
    df_filtered_bike_type = df_quality[mask_bike_type]
    if len(df_filtered_bike_type) >= n:
        return df_filtered_bike_type.head(n).index.tolist()
    if len(df_quality) >= n:
        return df_quality.head(n).index.tolist()
    return df_quality.index.tolist()


def get_top_n_recommendations(bike_similarity_df: pd.DataFrame, bike_id: int, preference_mask: set, n: int = 16) -> list:
    """
    Returns the top n recommendations for a bike_id, given a bike_similarity_df
    Args:
        bike_similarity_df (pd.DataFrame): bike similarity df
        bike_id (int): bike_id to get recommendations for
        preference_mask (set): preference mask
        n (int): number of recommendations to return
    Returns:
        list: list of top n recommendations for bike_id, skipping the bike_id itself
    """
    # get n*10 and apply preference filter
    # does not ensure optimality but is computationally very efficient
    top_candidates = bike_similarity_df.loc[bike_id].squeeze().nsmallest(n * 10 + 1).index.tolist()
    recommendations = [candidate for candidate in top_candidates if candidate in preference_mask]
    recommendations = recommendations[:n]

    if len(recommendations) < n:
        recommendations = bike_similarity_df.loc[bike_id].squeeze().nsmallest(n + 1).index.tolist()
    return recommendations


def get_top_n_recommendations_prefiltered(
    bike_similarity_df: pd.DataFrame,
    preference_mask: set,
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
        preference_mask (set): bike indicies matching preferences
        df (pd.DataFrame): dataframe of bikes
        df_status_masked (pd.DataFrame): dataframe of bikes with the given status
        bike_id (int): bike_id to get recommendations for
        n (int): number of recommendations to return
    Returns:
        list: list of top n recommendations for bike_id, skipping the bike_id itself
    """
    # get the values of the prefilter_features for the bike_id
    prefilter_values = df.loc[bike_id, prefilter_features]
    df_filtered = bike_similarity_df.loc[bike_id, (df_status_masked[prefilter_features] == prefilter_values).values]
    # get n*10 and apply preference filter
    # does not ensure optimality but is computationally very efficient
    if len(df_filtered) > 1:
        # Get the top n*10 smallest values
        top_candidates = df_filtered.squeeze().nsmallest(n * 10).index.tolist()

        # Filter the candidates based on the preference mask
        recommendations = [candidate for candidate in top_candidates if candidate in preference_mask]

        # Return the top n filtered candidates
        return recommendations[:n]
    else:
        return []


def get_top_n_recommendations_mix(
    bike_id: int,
    preference_mask: set,
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
        preference_mask (set): bike indicies matching preferences
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
            top_n_recommendations_generic = get_top_n_recommendations(bike_similarity_df, bike_id, preference_mask, n)

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
                    top_n_recommendations = interveave(top_n_recommendations_prefiltered, top_n_recommendations_generic)

                else:
                    top_n_recommendations = top_n_recommendations_prefiltered + top_n_recommendations_generic

            else:
                top_n_recommendations = top_n_recommendations_generic

            # remove duplicates
            top_n_recommendations = list(dict.fromkeys(top_n_recommendations))

            # return the top n recommendations
            return top_n_recommendations[:n], error

    except Exception as e:
        error = str(e)
        return top_n_recommendations, error


def get_mask_continent(data_store_content, continent_id):
    """
    Generate a mask based on continent-specific logic.
    Parameters:
    - data_store_content: The data store containing preference data.
    - continent_id: The ID of the continent to apply specific logic.
    Returns:
    - A list representing the combined preference mask.
    """
    # Filter recommendations for general preferences
    condition = (("continent_id", lambda df: df["continent_id"] == continent_id),)
    mask = get_preference_mask_condition(data_store_content.df_preference, condition)
    # If US or UK, also allow non-ebikes from EU
    if continent_id in [4, 7]:
        ebike_condition = (
            ("continent_id", lambda df: df["continent_id"] == 1),
            ("motor", lambda df: df["motor"] == 0),
        )
        ebike_mask = get_preference_mask_condition(data_store_content.df_preference, ebike_condition)
        mask = mask + ebike_mask
    return mask


def get_user_preference_mask(data_store_content, user_id, strategy_name):
    """
    Generate a preference mask based on user-specific preferences.
    Parameters:
    - data_store_content: The data store containing user-specific preference data.
    - user_id: The ID of the user to apply specific preferences.
    - strategy_name: The name of the strategy to determine filtering logic.
    Returns:
    - A list representing the user-specific preference mask.
    """
    if user_id != 0 and user_id in data_store_content.df_preference_user.index:
        specific_user_preferences = data_store_content.df_preference_user[data_store_content.df_preference_user.index == user_id]
        combined_conditions = []
        for index, row in specific_user_preferences.iterrows():
            numeric_frame_size = get_numeric_frame_size(row["frame_size"])
            if strategy_name != "product_page":
                combined_condition = lambda df, max_price=row["max_price"], category_id=row[
                    "category_id"
                ], frame_size=numeric_frame_size: (
                    (df["price"] <= max_price)
                    & ((category_id == 0) | (df["bike_category_id"] == category_id))
                    & (
                        (frame_size == 1)
                        | ((df["frame_size_code"] >= frame_size - 3) & (df["frame_size_code"] <= frame_size + 3))
                    )
                )
            else:
                combined_condition = lambda df, max_price=row["max_price"], frame_size=numeric_frame_size: (
                    (df["price"] <= max_price)
                    & (
                        (frame_size == 1)
                        | ((df["frame_size_code"] >= frame_size - 3) & (df["frame_size_code"] <= frame_size + 3))
                    )
                )
            combined_conditions.append(combined_condition)
        preference_user = (("combined", combined_conditions),)
        preference_mask_user = get_preference_mask_condition_list(data_store_content.df_preference, preference_user)
        return preference_mask_user
    return []
