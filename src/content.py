import pandas as pd

from typing import Tuple, List, Optional

from src.helper import interveave



def get_top_n_popularity_prefiltered(df_popularity: pd.DataFrame, family_id: int, price: int, frame_size_code: str, n: int = 16) -> list:
    """
    Returns the top n recommendations based on popularity, progressively filtering for price, frame_size_code, and family_id
    Args:
        df_popularity (pd.DataFrame): DataFrame with sorted bike ids by popularity
        family_id (int): family_id of the bike
        price (int): price of the bike
        frame_size_code (str): frame_size_code of the bike
        n (int): number of recommendations to return
    Returns:
        list: list of top n bike ids by popularity
    """
    # Filter for 20% higher and lower price
    df_filtered_price = df_popularity[
        (df_popularity["price"] >= price * 0.8) &
        (df_popularity["price"] <= price * 1.2)
    ]
    # Filter for same frame_size_code
    df_filtered_size = df_filtered_price[
        df_filtered_price["frame_size_code"] == frame_size_code
    ]
    # Filter for same family_id
    df_filtered_family = df_filtered_size[
        df_filtered_size["family_id"] == family_id
    ]
    # Step-wise approach to get at least n elements
    if len(df_filtered_family) >= n:
        return df_filtered_family.head(n).index.tolist()
    elif len(df_filtered_size) >= n:
        return df_filtered_size.head(n).index.tolist()
    elif len(df_filtered_price) >= n:
        return df_filtered_price.head(n).index.tolist()
    else:
        return df_popularity.head(n).index.tolist()


def get_top_n_recommendations(similarity_matrix: pd.DataFrame, bike_id: int, n: int = 16) -> list:
    """
    Returns the top n recommendations for a bike_id, given a similarity_matrix
    Args:
        similarity_matrix (pd.DataFrame): similarity matrix
        bike_id (int): bike_id to get recommendations for
        n (int): number of recommendations to return

    Returns:
        list: list of top n recommendations for bike_id, skipping the bike_id itself


    """

    # squeeze to convert pd.DataFrame into pd.Series
    return similarity_matrix.loc[bike_id].squeeze().nsmallest(n + 1).index.tolist()


def get_top_n_recommendations_prefiltered(
    similarity_matrix: pd.DataFrame, df: pd.DataFrame, df_status_masked: pd.DataFrame, bike_id: int, prefilter_features: list, logger, n: int = 16,
) -> list:
    """
    Returns the top n recommendations for a bike_id with a prefilter applied for the prefilter_features of the specific bike_id
    empty list if only the bike_id itself or less match the prefilter
    Args:
        similarity_matrix (pd.DataFrame): cosine similarity matrix
        df (pd.DataFrame): dataframe of bikes
        df_status_masked (pd.DataFrame): dataframe of bikes with the given status
        bike_id (int): bike_id to get recommendations for
        n (int): number of recommendations to return
    Returns:
        list: list of top n recommendations for bike_id, skipping the bike_id itself
    """

    # get the values of the prefilter_features for the bike_id
    prefilter_values = df.loc[bike_id, prefilter_features]

    # get nsmallest similarity_matrix index for the bike_ids that match prefilter_values
    # return empty list if prefilter is only returning the bike_id itself
    # if there are multiple prefilter_features
    #similarity_matrix.loc[bike_id, (df[prefilter_features] == prefilter_values).all(axis=1)]
    #similarity_matrix.loc[bike_id, (df[prefilter_features] == prefilter_values).any(axis=1)]



    if len(similarity_matrix.loc[bike_id, (df_status_masked[prefilter_features] == prefilter_values).values]) > 1:
        return(
            similarity_matrix.loc[bike_id, (df_status_masked[prefilter_features] == prefilter_values).values]
            .squeeze()
            .nsmallest(n)
            .index.tolist()
            )
    else:
        return([])

def get_top_n_recommendations_mix(
    bike_id: int,
    family_id: int,
    price: int,
    frame_size_code: str,
    df: pd.DataFrame,
    df_status_masked: pd.DataFrame,
    df_popularity: pd.DataFrame,
    similarity_matrix: pd.DataFrame,
    prefilter_features: list,
    logger,
    n: int = 16,
    ratio: float = 0.75,
    interveave_prefilter_general: bool = True,
) -> Tuple[List, Optional[str]]:
    """
    Mix of popularity and content based recommendations

    Logic:
        1. get the top n recommendations based on popularity, return if bike_id not in the df
        2. get the top n recommendations prefiltered by the prefilter_features for n * ratio
        3. get the top n recommendations for the bike_id for n
        4. interveave or append the lists in the order of 2, 3 and append 1; ensuring that enough recommendations are returned

    Args:
        bike_id (int): bike_id to get recommendations for
        df (pd.DataFrame): dataframe of bikes
        df_status_masked (pd.DataFrame): dataframe of bikes with the given status
        df_popularity (pd.DataFrame): dataframe of bikes sorted by popularity
        similarity_matrix (pd.DataFrame): similarity matrix
        prefilter_features (list): list of features to prefilter by
        n (int): number of recommendations to return
        interveave_prefilter_general (bool): if True, interveave prefiltered and generic recommendations, else append
        ratio (float): ratio of prefiltered recommendations to generic recommendations
        interveave_prefilter_general (bool): if True, interveave prefiltered and generic recommendations, else append


    Returns:
        tuple: list of top n recommendations for bike_id, error message


    """
    error = None
    top_n_recommendations = []

    try:
        if bike_id not in df.index:
            logger.warning("bike_id not in df, using popularity recommendations",
                extra={
                    "bike_id": bike_id,
                })

            top_n_popularity = get_top_n_popularity_prefiltered(df_popularity, family_id, price, frame_size_code, n)
            return top_n_popularity[:n], error

        else:
            # prefiltered recommendations
            top_n_recommendations_prefiltered = get_top_n_recommendations_prefiltered(
                similarity_matrix, df, df_status_masked, bike_id, prefilter_features, logger, int(n * ratio)
            )

            # get the top n recommendations for the bike_id
            top_n_recommendations_generic = get_top_n_recommendations(similarity_matrix, bike_id, n)



            #remove bike_id from recommendations, we do not want to recommend the same bike
            try:
                top_n_recommendations_generic.remove(bike_id)
                top_n_recommendations_prefiltered.remove(bike_id)
            except:
                pass


            # if prefiltered recommendations exist
            if top_n_recommendations_prefiltered:


                # interveave prefiltered and generic recommendations
                if interveave_prefilter_general:
                    top_n_recommendations = interveave(top_n_recommendations_prefiltered, top_n_recommendations_generic)

                else: top_n_recommendations = top_n_recommendations_prefiltered + top_n_recommendations_generic

            else: top_n_recommendations = top_n_recommendations_generic




            # remove duplicates
            top_n_recommendations = list(dict.fromkeys(top_n_recommendations))


            # return the top n recommendations
            return top_n_recommendations[:n], error

    except Exception as e:
        error = str(e)
        return top_n_recommendations, error
