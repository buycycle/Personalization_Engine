"""Personalization Engine"""
import random
from fastapi import Body
from pydantic import BaseModel, validator
import os
import time

# fastapi
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

# periodical data read-in
from threading import Thread

# config file
import configparser

# get loggers
from buycycle.logger import Logger
from buycycle.data import get_numeric_frame_size, get_preference_mask

# sql queries and feature selection
from src.driver_content import prefilter_features
from src.driver_collaborative import (
    bike_id,
    features,
    item_features,
    user_features,
    implicit_feedback,
)

# import functions from src folder
from src.data_content import DataStoreContent
from src.collaborative import DataStoreCollaborative
from src.strategies import (
    StrategyFactory,
    FallbackContentMixed,
    ContentMixed,
    Collaborative,
    CollaborativeRandomized,
    QualityFilter,
)
from src.strategies import strategy_dict

config_paths = "config/config.ini"
config = configparser.ConfigParser()
config.read(config_paths)
path = "data/"

app = FastAPI()

# read the environment from the docker environment variable
environment = os.getenv("ENVIRONMENT")
ab = os.getenv("AB")
app_name = "recommender-system"
app_version = "canary-008-bot"

logger = Logger.configure_logger(environment, ab, app_name, app_version)
logger.info("FastAPI app started")

# create data store
data_store_content = DataStoreContent(prefilter_features=prefilter_features)
data_store_collaborative = DataStoreCollaborative()
data_store_content_available = False
data_store_collaborative_available = False

# inital data readin
while True:
    try:
        data_store_content.read_data()
        data_store_content_available = True
        break
    except Exception as e:
        logger.error(
            f"Content data could not initially be read, error: {e}. Trying again in 60 seconds."
        )
        time.sleep(60)

while True:
    try:
        data_store_collaborative.read_data()
        data_store_collaborative_available = True
        break
    except Exception as e:
        logger.error(
            f"Collaborative data could not initially be read, error: {e}. Trying again in 60 seconds."
        )
        time.sleep(60)

read_interval = 60 + random.uniform(-5, 5)
# read the data periodically
data_loader_content = Thread(
    target=data_store_content.read_data_periodically, args=(read_interval, logger)
)
data_loader_collaborative = Thread(
    target=data_store_collaborative.read_data_periodically, args=(read_interval, logger)
)

data_loader_content.start()
data_loader_collaborative.start()


@app.get("/")
def home():
    return {"message": "Recommender system"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/health_model")
def health_model_check():
    if data_store_content_available and data_store_collaborative_available:
        return {"status": "ok"}
    else:
        # Return a 503 Service Unavailable status code with a message
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "message": "One or more data stores are not loaded with data.",
            },
        )


class RecommendationRequest(BaseModel):
    user_id: int = 0
    distinct_id: str = "NA"
    continent_id: int = 1
    bike_id: int = 0
    bike_type: int = 1
    family_id: int = 1101
    price: int = 1200
    frame_size_code: str = "56"
    rider_height: int = 180
    category: str = "road"
    n: int = 12
    strategy: str = "product_page"

    @validator("user_id", pre=True)
    def validate_user_id(cls, value):
        if value is None:
            return 0
        if isinstance(value, str):
            try:
                # Attempt to convert the string to an integer
                return int(value)
            except ValueError:
                # If conversion fails, return the default value
                return 0
        return value


@app.post("/recommendation")
def recommendation(request_data: RecommendationRequest = Body(...)):
    user_id = request_data.user_id
    distinct_id = request_data.distinct_id
    continent_id = request_data.continent_id
    bike_id = request_data.bike_id
    bike_type = request_data.bike_type
    family_id = request_data.family_id
    price = request_data.price
    frame_size_code = get_numeric_frame_size(
        request_data.frame_size_code, bike_type, default_value=56
    )
    rider_height = request_data.rider_height
    category = request_data.category
    n = request_data.n
    strategy_name = request_data.strategy

    # the user_id the model is trained on is a mix of user_id and distinct_id
    # check here if the query containes a user_id
    # if yes, base the recommendation on the user_id
    # if not, base the recommendation on the distinct_id

    if user_id == 0:
        id = distinct_id
    else:
        id = str(user_id)

    # randomize over the top n * x
    sample = n * 5

    # Instantiate strategy
    strategy_target = strategy_name

    # lock the data stores to prevent data from being updated while we are using it
    with data_store_collaborative._lock and data_store_content._lock:
        # filter recommendations for preferences
        preferences = {
            "continent_id": continent_id,
        }
        preference_mask = get_preference_mask(
            data_store_content.df_preference, preferences
        )

        # if US or UK, also allow non-ebikes from EU
        # merge with preferences above with OR condition
        if continent_id in [4, 7]:
            ebike_sending_preferences = {
                "continent_id": 1,
                "motor": 0,
            }
            ebike_preference_mask = get_preference_mask(
                data_store_content.df_preference, ebike_sending_preferences
            )

            preference_mask = preference_mask + ebike_preference_mask

        strategy_factory = StrategyFactory(strategy_dict)

        # if strategy_name not in list, use FallbackContentMixed
        strategy_instance = strategy_factory.get_strategy(
            strategy_name=strategy_name,
            fallback_strategy=CollaborativeRandomized,
            logger=logger,
            data_store_collaborative=data_store_collaborative,
            data_store_content=data_store_content,
        )
        # raise error if we do not know strategy
        if strategy_instance is None:
            accepted_strategies = list(strategy_dict.keys())
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown strategy. Accepted strategies are: {accepted_strategies}. "
                "If the strategy field is empty, it will default to 'product_page'.",
            )

        # Recommend
        # different strategies use different inputs, think about how to clean this up
        if isinstance(strategy_instance, (ContentMixed, FallbackContentMixed)):
            strategy, recommendation, error = strategy_instance.get_recommendations(
                bike_id,
                preference_mask,
                bike_type,
                family_id,
                price,
                frame_size_code,
                n,
            )
        elif isinstance(strategy_instance, Collaborative):
            strategy, recommendation, error = strategy_instance.get_recommendations(
                id, preference_mask, n
            )
        elif isinstance(strategy_instance, CollaborativeRandomized):
            strategy, recommendation, error = strategy_instance.get_recommendations(
                id, preference_mask, n, sample
            )
        elif isinstance(strategy_instance, QualityFilter):
            strategy, recommendation, error = strategy_instance.get_recommendations(
                category,
                price,
                rider_height,
                preference_mask,
                n,
            )
        else:
            # Handle unknown strategy
            accepted_strategies = list(strategy_dict.keys())
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown strategy. Accepted strategies are: {accepted_strategies}. "
                "If the strategy field is empty, it will default to 'product_page'.",
            )

        # Fall back strategy if not enough recommendations were generated for the product_page
        if len(recommendation) != n and strategy_name == "product_page":
            logger.error(
                f"Not enough recommendations generated by strategy '{strategy_name}'. "
                f"Expected: {n}, produced: {len(recommendation)}. Using fallback_strategy 'FallbackContentMixed'."
            )
            strategy_name = "FallbackContentMixed"
            strategy_instance = strategy_factory.get_strategy(
                strategy_name=strategy_name,
                fallback_strategy=FallbackContentMixed,
                logger=logger,
                data_store_collaborative=data_store_collaborative,
                data_store_content=data_store_content,
            )
            strategy, recommendation, error = strategy_instance.get_recommendations(
                bike_id,
                preference_mask,
                bike_type,
                family_id,
                price,
                frame_size_code,
                n,
            )

        logger.info(
            "successful recommendation",
            extra={
                "strategy_target": strategy_target,
                "strategy_used": strategy,
                "user_id": user_id,
                "distinct_id": distinct_id,
                "continent_id": continent_id,
                "bike_id": bike_id,
                "bike_type": bike_type,
                "family_id": family_id,
                "price": price,
                "frame_size_code": frame_size_code,
                "n": n,
                "recommendation": recommendation,
            },
        )

        if error:
            # Return error response if it exists
            logger.error("Error no recommendation available, exception: " + error)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Error no recommendation available",
            )
        else:
            # Return success response with recommendation data
            return {
                "status": "success",
                "strategy": strategy,
                "recommendation": recommendation,
                "app_name": app_name,
                "app_version": app_version,
            }


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Log the error details using the provided logger
    logger.error(
        "400 Bad Request: " + str(exc),
        extra={
            "info": "Bad request",
        },
    )
    # Construct a hint for the expected request body format
    expected_format = RecommendationRequest.schema()
    # Return a JSON response with the error details and the hint
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "status": "error",
            "message": "Bad Request: " + str(exc),
            "hint": "Expected request body format",
            "expected_format": expected_format,
        },
    )


# Error handling for 500 Internal Server Error
@app.exception_handler(500)
def internal_server_error_handler(request: Request, exc: HTTPException):
    # Log the error details using the provided logger
    logger.error(
        "500 Internal Server Error: " + str(exc),
        extra={
            "info": "Internal server error",
        },
    )
    # Return a JSON response with the error details
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"status": "error", "message": "Internal Server Error: " + str(exc)},
    )
