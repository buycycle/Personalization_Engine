import pandas as pd


from abc import ABC, abstractmethod

from src.content import get_top_n_recommendations
from src.content import get_top_n_recommendations_prefiltered
from src.content import get_top_n_quality_prefiltered
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
        self, bike_id: int, family_id: int, price: int, frame_size_code: str, n: int
    ) -> Tuple[str, List, Optional[str]]:
        bike_similarity_df, error = construct_dense_similarity_row(self.similarity_matrix, bike_id)

        recommendations, error = get_top_n_recommendations_mix(
            bike_id,
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
        self, bike_id: int, family_id: int, price: int, frame_size_code: str, n: int
    ) -> Tuple[str, List, Optional[str]]:
        bike_similarity_df, error = construct_dense_similarity_row(self.similarity_matrix, bike_id)

        recommendations, error = get_top_n_recommendations_mix(
            bike_id,
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

    def get_recommendations(self, user_id: str, n: int) -> Tuple[str, List, Optional[str]]:
        recommendations, error = get_top_n_collaborative(
            self.model,
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

    def get_recommendations(self, user_id: str, n: int, sample: int) -> Tuple[str, List, Optional[str]]:
        recommendations, error = get_top_n_collaborative_randomized(
            self.model,
            user_id,
            n,
            sample,
            self.dataset,
            self.df_status_masked,
            self.logger,
        )
        return self.strategy, recommendations, error


# untested
class CollaborativeRandomizedContentInterveaved(RecommendationStrategy):
    """Collaborative filtering with randomized sampling interweaved with content-based recommendations."""

    def __init__(self, logger, data_store_collaborative, data_store_content):
        self.strategy = "CollaborativeRandomizedContentInterveaved"
        self.collaborative_model = data_store_collaborative.model
        self.collaborative_dataset = data_store_collaborative.dataset
        self.content_df = data_store_content.df
        self.content_df_status_masked = data_store_content.df_status_masked
        self.content_df_quality = data_store_content.df_quality
        self.content_similarity_matrix = data_store_content.similarity_matrix
        self.content_prefilter_features = data_store_content.prefilter_features
        self.logger = logger

    def get_recommendations(
        self, user_id: str, bike_id: int, family_id: int, price: int, frame_size_code: str, n: int, sample: int
    ) -> Tuple[str, List, Optional[str]]:
        # Get collaborative recommendations with randomized sampling
        collaborative_recommendations, collaborative_error = get_top_n_collaborative_randomized(
            self.collaborative_model,
            user_id,
            n,
            sample,
            self.collaborative_dataset,
            self.content_df_status_masked,
            self.logger,
        )

        # Get content-based recommendations
        bike_similarity_df, content_error = construct_dense_similarity_row(self.content_similarity_matrix, bike_id)
        content_recommendations, content_error = get_top_n_recommendations_mix(
            bike_id,
            family_id,
            price,
            frame_size_code,
            self.content_df,
            self.content_df_status_masked,
            self.content_df_quality,
            bike_similarity_df,
            self.content_prefilter_features,
            self.logger,
            n,
            ratio=0.5,
            interveave_prefilter_general=True,
        )

        # Interveave collaborative and content recommendations using the provided function
        interveaved_recommendation = interveave(collaborative_recommendations, content_recommendations)

        # Truncate the list to the desired number of recommendations
        interveaved_recommendation = interveaved_recommendation[:n]

        # Combine errors if any
        error = collaborative_error or content_error
        return self.strategy, interveaved_recommendation, error


# Dictionary of strategies
strategy_dict = {
    "product_page": ContentMixed,
    "braze": Collaborative,
    "homepage": CollaborativeRandomized,
    "FallbackContentMixed": FallbackContentMixed,
}
# Instantiate the StrategyFactory with the dictionary
strategy_factory = StrategyFactory(strategy_dict)
