import pandas as pd


from abc import ABC, abstractmethod

from src.content import get_top_n_recommendations
from src.content import get_top_n_recommendations_prefiltered
from src.content import get_top_n_quality_prefiltered_bot
from src.content import get_top_n_recommendations_mix

from src.collaborative import (
    get_top_n_collaborative_randomized,
    get_top_n_collaborative_rerank,
    read_data_model,
)
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
        self,
        bike_id: int,
        preference_mask: list,
        bike_type: int,
        family_id: int,
        price: int,
        frame_size_code: str,
        n: int,
    ) -> Tuple[str, List, Optional[str]]:
        # xxx do we make sure that the bike_id is in the similarity_matrix?
        bike_similarity_df, error = construct_dense_similarity_row(
            self.similarity_matrix, bike_id
        )

        filter_features = (
            ("bike_type", lambda df: df["bike_type"] == bike_type),
            (
                "price",
                lambda df: (df["price"] >= price * 0.8) & (df["price"] <= price * 1.2),
            ),
            ("frame_size_code", lambda df: df["frame_size_code"] == frame_size_code),
            ("family_id", lambda df: df["family_id"] == family_id),
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
        self,
        bike_id: int,
        preference_mask: set,
        bike_type: int,
        family_id: int,
        price: int,
        frame_size_code: str,
        n: int,
    ) -> Tuple[str, List, Optional[str]]:
        # xxx do we make sure that the bike_id is in the similarity_matrix?
        bike_similarity_df, error = construct_dense_similarity_row(
            self.similarity_matrix, bike_id
        )
        filter_features = (
            ("bike_type", lambda df: df["bike_type"] == bike_type),
            (
                "price",
                lambda df: (df["price"] >= price * 0.8) & (df["price"] <= price * 1.2),
            ),
            ("frame_size_code", lambda df: df["frame_size_code"] == frame_size_code),
            ("family_id", lambda df: df["family_id"] == family_id),
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
        self.strategy = "CollaborativeStrategyRandomized"
        self.model = data_store_collaborative.model
        self.dataset = data_store_collaborative.dataset
        self.df_status_masked = data_store_content.df_status_masked
        self.logger = logger

    def get_recommendations(
        self, user_id: str, preference_mask: set, n: int, sample: int
    ) -> Tuple[str, List, Optional[str]]:
        df_status_masked_set = set(self.df_status_masked.index)

        recommendations, error = get_top_n_collaborative_randomized(
            self.model,
            user_id,
            preference_mask,
            n,
            sample,
            self.dataset,
            df_status_masked_set,
            self.logger,
        )
        return self.strategy, recommendations, error


class CollaborativeRandomized(RecommendationStrategy):
    """Collaborative filtering with randomized sampling"""

    def __init__(self, logger, data_store_collaborative, data_store_content):
        self.strategy = "CollaborativeStrategyRandomized"
        self.model = data_store_collaborative.model
        self.dataset = data_store_collaborative.dataset
        self.df_status_masked = data_store_content.df_status_masked
        self.logger = logger

    def get_recommendations(
        self, user_id: str, preference_mask: set, n: int, sample: int
    ) -> Tuple[str, List, Optional[str]]:
        df_status_masked_set = set(self.df_status_masked.index)

        recommendations, error = get_top_n_collaborative_randomized(
            self.model,
            user_id,
            preference_mask,
            n,
            sample,
            self.dataset,
            df_status_masked_set,
            self.logger,
        )
        return self.strategy, recommendations, error


class CollaborativeRerank(RecommendationStrategy):
    """Try Collaborative filtering, fail silently and return an empty list"""

    def __init__(self, logger, data_store_collaborative, data_store_content):
        self.strategy = "CollaborativeRerank"
        self.model = data_store_collaborative.model
        self.dataset = data_store_collaborative.dataset
        self.logger = logger

    def get_recommendations(
        self, user_id: str, bike_rerank_id: list
    ) -> Tuple[str, List, Optional[str]]:
        recommendations, error = get_top_n_collaborative_rerank(
            self.model,
            user_id,
            bike_rerank_id,
            self.dataset,
            self.logger,
        )
        return self.strategy, recommendations, error


class QualityFilter(RecommendationStrategy):
    """Apply filters and sort by quality score"""

    def __init__(self, logger, data_store_collaborative, data_store_content):
        self.strategy = "QualityFilter"
        self.df_quality = data_store_content.df_quality

    def get_recommendations(
        self,
        category: str,
        price: int,
        rider_height: int,
        is_ebike: int,
        is_frameset: int,
        brand: str,
        preference_mask: set,
        n: int,
    ) -> Tuple[str, List[int], Optional[str]]:
        # Define the quality_features tuple with filter conditions
        quality_features = [
            ("category", lambda df: df["category"] == category),
            ("rider_height_max", lambda df: df["rider_height_max"] >= rider_height),
            ("rider_height_min", lambda df: df["rider_height_min"] <= rider_height),
            (
                "price",
                lambda df: (df["price"] >= price * 0.8) & (df["price"] <= price * 1.2),
            ),
            ("is_ebike", lambda df: df["is_ebike"] == is_ebike),
            ("is_frameset", lambda df: df["is_frameset"] == is_frameset),
        ]
        # Only add the brand filter if brand is not "null"
        if brand != "null":
            quality_features.append(("brand", lambda df: df["brand"] == brand))
        # Convert the list to a tuple if necessary
        quality_features = tuple(quality_features)
        recommendations, error = get_top_n_quality_prefiltered_bot(
            self.df_quality, preference_mask, quality_features, n
        )
        return self.strategy, recommendations, error


# Dictionary of strategies
strategy_dict = {
    "product_page": ContentMixed,
    "braze": Collaborative,
    "homepage": CollaborativeRandomized,
    "rerank": CollaborativeRerank,
    "FallbackContentMixed": FallbackContentMixed,
    "bot": QualityFilter,
}
# Instantiate the StrategyFactory with the dictionary
strategy_factory = StrategyFactory(strategy_dict)
