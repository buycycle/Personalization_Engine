import pandas as pd


from abc import ABC, abstractmethod

from src.content import get_top_n_recommendations
from src.content import get_top_n_recommendations_prefiltered
from src.content import get_top_n_quality_prefiltered_bot
from src.content import get_top_n_recommendations_mix

from src.collaborative import get_top_n_collaborative, get_top_n_collaborative_randomized, read_data_model
from src.data_content import construct_dense_similarity_row

from src.helper import interveave
from typing import Union, List, Tuple, Optional


class RecommendationStrategy(ABC):
    @abstractmethod
    def get_recommendations(self, bike_id: int, n: int) -> Union[str, List, str]:
        pass


class StrategyFactory:
    def __init__(self, strategies, **kwargs):
        self._strategies = strategies

    def get_strategy(self, strategy_name, fallback_strategy=None, **kwargs):
        strategy_class = self._strategies.get(strategy_name)
        if strategy_class:
            return strategy_class(**kwargs)
        elif fallback_strategy:
            return fallback_strategy(**kwargs)
        raise ValueError(f"Unknown strategy and no fallback defined {strategy_name}")


class FallbackContentMixed(RecommendationStrategy):
    def __init__(self, logger, data_store_collaborative, data_store_content):
        self.strategy = "FallbackContentMixed"
        self.df = data_store_content.df
        self.df_status_masked = data_store_content.df_status_masked
        self.df_quality = data_store_content.df_quality
        self.similarity_matrix = data_store_content.similarity_matrix
        self.prefilter_features = data_store_content.prefilter_features
        self.logger = logger

    def get_recommendations(
            self, bike_id: int, preference_mask: list, bike_type: int, family_id: int, price: int, frame_size_code: str, n: int
    ) -> Tuple[str, List, Optional[str]]:
        bike_similarity_df, error = construct_dense_similarity_row(self.similarity_matrix, bike_id)

        filter_features = (
            ("bike_type", lambda df: df["bike_type"] == bike_type),
            ("price", lambda df: (df["price"] >= price * 0.8) & (df["price"] <= price * 1.2)),
            ("frame_size_code", lambda df: df["frame_size_code"] == frame_size_code),
            ("family_id", lambda df: df["family_id"] == family_id)
        )

        recommendations, error = get_top_n_recommendations_mix(
            bike_id,
            preference_mask,
            filter_features,
            bike_type,
            family_id,
            price,
            frame_size_code,
            self.df,
            self.df_status_masked,
            self.df_quality,
            bike_similarity_df,
            self.prefilter_features,
            self.logger,
            n,
            ratio=0.5,
            interveave_prefilter_general=False,
        )
        return self.strategy, recommendations, error


class ContentMixed(RecommendationStrategy):
    """Content based according to prefiltered mix with generic, returns quality if too little specific
    recommendations are available"""

    def __init__(self, logger, data_store_collaborative, data_store_content):
        self.strategy = "ContentMixed"
        self.df = data_store_content.df
        self.df_status_masked = data_store_content.df_status_masked
        self.df_quality = data_store_content.df_quality
        self.similarity_matrix = data_store_content.similarity_matrix
        self.prefilter_features = data_store_content.prefilter_features
        self.logger = logger

    def get_recommendations(
            self, bike_id: int, preference_mask: list, bike_type: int, family_id: int, price: int, frame_size_code: str, n: int
    ) -> Tuple[str, List, Optional[str]]:
        bike_similarity_df, error = construct_dense_similarity_row(self.similarity_matrix, bike_id)
        filter_features = (
            ("bike_type", lambda df: df["bike_type"] == bike_type),
            ("price", lambda df: (df["price"] >= price * 0.8) & (df["price"] <= price * 1.2)),
            ("frame_size_code", lambda df: df["frame_size_code"] == frame_size_code),
            ("family_id", lambda df: df["family_id"] == family_id)
        )

        recommendations, error = get_top_n_recommendations_mix(
            bike_id,
            preference_mask,
            filter_features,
            bike_type,
            family_id,
            price,
            frame_size_code,
            self.df,
            self.df_status_masked,
            self.df_quality,
            bike_similarity_df,
            self.prefilter_features,
            self.logger,
            n,
            ratio=0.5,
            interveave_prefilter_general=False,
        )
        return self.strategy, recommendations, error


class Collaborative(RecommendationStrategy):
    """Try Collaborative filtering, fail silently and return an empty list"""

    def __init__(self, logger, data_store_collaborative, data_store_content):
        self.strategy = "Collaborative"
        self.model = data_store_collaborative.model
        self.dataset = data_store_collaborative.dataset
        self.df_status_masked = data_store_content.df_status_masked
        self.logger = logger

    def get_recommendations(self, user_id: str, preference_mask: list, n: int) -> Tuple[str, List, Optional[str]]:
        recommendations, error = get_top_n_collaborative(
            self.model,
            preference_mask,
            user_id,
            n,
            self.dataset,
            self.df_status_masked,
            self.logger,
        )
        return self.strategy, recommendations, None


class CollaborativeRandomized(RecommendationStrategy):
    """Collaborative filtering with randomized sampling"""

    def __init__(self, logger, data_store_collaborative, data_store_content):
        self.strategy = "CollaborativeStrategyRandomized"
        self.model = data_store_collaborative.model
        self.dataset = data_store_collaborative.dataset
        self.df_status_masked = data_store_content.df_status_masked
        self.logger = logger

    def get_recommendations(self, user_id: str, preference_mask: list, n: int, sample: int) -> Tuple[str, List, Optional[str]]:



        preference_mask_set = set(preference_mask)
        df_status_masked_set = set(self.df_status_masked.index)

        recommendations, error = get_top_n_collaborative_randomized(
            self.model,
            user_id,
            preference_mask_set,
            n,
            sample,
            self.dataset,
            df_status_masked_set,
            self.logger,
        )
        return self.strategy, recommendations, error

class QualityFilter(RecommendationStrategy):
    """Apply filters and sort by quality score"""
    def __init__(self, logger, data_store_collaborative, data_store_content):
        self.strategy = "QualityFilter"
        self.df_quality = data_store_content.df_quality
    def get_recommendations(self, bike_type: int, price: int, rider_height_max: int, rider_height_min: int, family_id: int, preference_mask: List[int], n: int) -> Tuple[str, List[int], Optional[str]]:
        preference_mask_set = set(preference_mask)
        # Define the quality_features tuple with filter conditions
        filter_features = (
            ("bike_type", lambda df: df["bike_type"] == bike_type),
            ("price", lambda df: (df["price"] >= price * 0.8) & (df["price"] <= price * 1.2)),
            ("rider_height_max", lambda df: df["rider_height_max"] <= rider_height_max),
            ("rider_height_min", lambda df: df["rider_height_min"] >= rider_height_min),
            ("family_id", lambda df: df["family_id"] == family_id)
        )
        recommendations, error = get_top_n_quality_prefiltered_bot(
            self.df_quality,
            preference_mask_set,
            filter_features,
            n
        )
        return self.strategy, recommendations, error


# Dictionary of strategies
strategy_dict = {
    "product_page": ContentMixed,
    "braze": Collaborative,
    "homepage": CollaborativeRandomized,
    "FallbackContentMixed": FallbackContentMixed,
    "bot": QualityFilter,
}
# Instantiate the StrategyFactory with the dictionary
strategy_factory = StrategyFactory(strategy_dict)
