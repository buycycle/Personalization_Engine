"""recommender system"""
# get env variable
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
from buycycle.logger import KafkaLogger

# sql queries and feature selection
from src.driver_content import prefilter_features
from src.driver_collaborative import bike_id, features, item_features, user_features, implicit_feedback

# import functions from src folder
from src.data_content import DataStoreContent
from src.collaborative import DataStoreCollaborative
from src.strategies import StrategyFactory, FallbackContentMixed, ContentMixed, Collaborative, CollaborativeRandomized, CollaborativeRandomizedContentInterveaved
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
app_version = "canary-003"

KAFKA_TOPIC = config["KAFKA"]["topic"]
KAFKA_BROKER = config["KAFKA"]["broker"]
logger = Logger.configure_logger(environment, ab, app_name, app_version)
logger = KafkaLogger(environment, ab, app_name, app_version, KAFKA_TOPIC, KAFKA_BROKER)
logger.info("FastAPI app started")

# create data store
data_store_content = DataStoreContent(prefilter_features=prefilter_features)
data_store_collaborative = DataStoreCollaborative()

# inital data readin
while True:
    try:
        data_store_content.read_data()
        break
    except Exception as e:
        logger.error(f"Content data could not initially be read, error: {e}. Trying again in 60 seconds.")
        time.sleep(60)

while True:
    try:
        data_store_collaborative.read_data()
        break
    except Exception as e:
        logger.error(f"Collaborative data could not initially be read, error: {e}. Trying again in 60 seconds.")
        time.sleep(60)

# read the data periodically
data_loader_content = Thread(target=data_store_content.read_data_periodically, args=(20, logger))
data_loader_collaborative = Thread(target=data_store_collaborative.read_data_periodically, args=(20, logger))

data_loader_content.start()
data_loader_collaborative.start()


@app.get("/")
def home():
    return {"message": "Recommender system"}


class RecommendationRequest(BaseModel):
    user_id: int = 0
    distinct_id: str = "NA"
    bike_id: int = 0
    family_id: int = 1101
    price: int = 1200
    frame_size_code: str = "56"
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
    bike_id = request_data.bike_id
    family_id = request_data.family_id
    price = request_data.price
    frame_size_code = request_data.frame_size_code
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

    # randomize over the top 100
    sample = 100

    # Instantiate strategy
    # Assuming this is what you meant by 'strategy_traget'
    strategy_target = strategy_name

    # lock the data stores to prevent data from being updated while we are using it
    with data_store_collaborative._lock and data_store_content._lock:
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
            strategy, recommendation, error = strategy_instance.get_recommendations(bike_id, family_id, price, frame_size_code, n)
        elif isinstance(strategy_instance, Collaborative):
            strategy, recommendation, error = strategy_instance.get_recommendations(id, n)
        elif isinstance(strategy_instance, CollaborativeRandomized):
            strategy, recommendation, error = strategy_instance.get_recommendations(id, n, sample)
        elif isinstance(strategy_instance, CollaborativeRandomizedContentInterveaved):
            # Ensure that the additional parameters required for this strategy are available
            strategy, recommendation, error = strategy_instance.get_recommendations(id, bike_id, family_id, price, frame_size_code, n, sample)
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
            strategy_name = "FallbackContentMixed"
            strategy_instance = strategy_factory.get_strategy(
                strategy_name=strategy_name,
                fallback_strategy=FallbackContentMixed,
                logger=logger,
                data_store_collaborative=data_store_collaborative,
                data_store_content=data_store_content,
            )
            strategy, recommendation, error = strategy_instance.get_recommendations(bike_id, family_id, price, frame_size_code, n)

    # Convert the recommendation to int if recommendation is not an empty list
    if len(recommendation) > 0:
        recommendation = [int(i) for i in recommendation]
    logger.info(
        "successful recommendation",
        extra={
            "strategy_target": strategy_target,
            "strategy_used": strategy,
            "user_id": user_id,
            "distinct_id": distinct_id,
            "bike_id": bike_id,
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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Error no recommendation available")
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
