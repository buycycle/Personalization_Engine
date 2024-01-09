"""recommender system"""
# get env variable
import os
import time
# flask
from flask import Flask, request, jsonify

# periodical data read-in
from threading import Thread

import pandas as pd

# config file
import configparser

# get loggers
from buycycle.logger import Logger
from buycycle.logger import KafkaLogger


# sql queries and feature selection
from src.driver_content import prefilter_features
from src.driver_collaborative import user_id, bike_id, features, item_features, user_features, implicit_feedback

# import functions from src folder
from src.data_content import DataStoreContent
from src.collaborative import DataStoreCollaborative
from src.strategies import StrategyFactory, FallbackContentMixed, ContentMixed, Collaborative, CollaborativeRandomized

from src.helper import get_field_value

config_paths = "config/config.ini"

config = configparser.ConfigParser()
config.read(config_paths)

path = "data/"

app = Flask(__name__)
# read the environment from the docker environment variable
environment = os.getenv("ENVIRONMENT")
ab = os.getenv("AB")
app_name = "recommender-system"
app_version = 'stable-001'

KAFKA_TOPIC = config["KAFKA"]["topic"]
KAFKA_BROKER = config["KAFKA"]["broker"]

logger = Logger.configure_logger(environment, ab, app_name, app_version)
logger = KafkaLogger(environment, ab, app_name,
                     app_version, KAFKA_TOPIC, KAFKA_BROKER)

logger.info("Flask app started")

# create data stores
data_store_content = DataStoreContent(prefilter_features=prefilter_features)
data_store_collaborative = DataStoreCollaborative()

# inital data readin
while True:
    try:
        data_store_content.read_data()
        break
    except Exception as e:
        logger.error("Content data could not initially be red, trying in 60sec")
        time.sleep(60)

while True:
    try:
        data_store_collaborative.read_data()
        break
    except Exception as e:
        logger.error(
            "Collaboration data could not initially be red, trying in 60sec")
        time.sleep(60)

# read the data periodically
data_loader_content = Thread(
    target=data_store_content.read_data_periodically, args=(20, logger))
data_loader_collaborative = Thread(
    target=data_store_collaborative.read_data_periodically, args=(20, logger))

data_loader_content.start()
data_loader_collaborative.start()


strategy_factory = StrategyFactory({
    'product_page': CollaborativeRandomized,
    'braze': Collaborative,
    'homepage': CollaborativeRandomized,
    'FallbackContentMixed': FallbackContentMixed,
})


@app.route("/")
def home():
    html = f"<h3>Recommender system</h3>"
    return html.format(format)


@app.route("/recommendation", methods=["POST"])
def recommendation():
    """take in user_id and bike_id and return a recommendation
    the payload should be in the following format:

    {
        "user_id": int,
        "distinct_id": str,
        "bike_id": int,
        "family_id": int,
        "price": int,
        "frame_size_code": str,
        "n": int
    }

    """

    # Logging the input payload
    json_payload = request.json

    recommendation_payload = pd.DataFrame(json_payload, index=[0])

    # read from request, assign default if missing or null
    user_id = get_field_value(recommendation_payload, "user_id", 0)

    distinct_id = get_field_value(
        recommendation_payload, "distinct_id", "NA", dtype=str)
    bike_id = get_field_value(recommendation_payload, "bike_id", 0)

    family_id = get_field_value(recommendation_payload, "family_id", 1101)
    price = get_field_value(recommendation_payload, "price", 1200)
    frame_size_code = get_field_value(
        recommendation_payload, "frame_size_code", "56", dtype=str)

    n = get_field_value(recommendation_payload, "n", 12)

    # randomize over the top 100
    sample = 100

    # instanciate strategy
    strategy_traget =  get_field_value(
        recommendation_payload, "strategy", "NA", dtype=str)

    # if empty use CollaborativeRandomized
    strategy_name = get_field_value(
        recommendation_payload, "strategy", "product_page", dtype=str)
    # if strategy_name not in list, use FallbackContentMixed
    strategy_instance = strategy_factory.get_strategy(
        strategy_name=strategy_name,
        # Specify the fallback strategy class here
        fallback_strategy=CollaborativeRandomized,
        logger=logger,
        data_store_collaborative=data_store_collaborative,
        data_store_content=data_store_content
    )

    # recommend
    # different strategies use different inputs, think about how to clean this up
    if isinstance(strategy_instance, (ContentMixed, FallbackContentMixed)):
        # If the strategy instance is either ContentMixed or FallbackContentMixed, apply the same logic
        strategy, recommendation, error = strategy_instance.get_recommendations(
            bike_id, family_id, price, frame_size_code, n)
    elif isinstance(strategy_instance, Collaborative):
        strategy, recommendation, error = strategy_instance.get_recommendations(
            distinct_id, n)
    elif isinstance(strategy_instance, CollaborativeRandomized):
        strategy, recommendation, error = strategy_instance.get_recommendations(
            distinct_id, n, sample)
    else:
        # Handle unknown strategy
        return jsonify({"status": "error", "message": "Unknown strategy"}), 400

    # fall back strategy if not enough recommendations were generated for the product_page
    if len(recommendation) != n and strategy_name == 'product_page':


        strategy_name = 'FallbackContentMixed'
        strategy_instance = strategy_factory.get_strategy(strategy_name=strategy_name,
                                                          # Specify the fallback strategy class here
                                                          fallback_strategy=FallbackContentMixed,
                                                          logger=logger,
                                                          data_store_collaborative=data_store_collaborative,
                                                          data_store_content=data_store_content)
        strategy, recommendation, error = strategy_instance.get_recommendations(
            bike_id, family_id, price, frame_size_code, n)

    # convert the recommendation to int
    recommendation = [int(i) for i in recommendation]

    logger.info(
                extra={
                    "strategy_traget": strategy_traget,
                    "strategy_used": strategy,
                    "user_id": user_id,
                    "distinct_id": distinct_id,
                    "bike_id": bike_id,
                    "family_id": family_id,
                    "price": price,
                    "frame_size_code": frame_size_code,
                    "n": n,
                    "recommendation": recommendation,
                })

    if error:
       # Return error response if it exists
        logger.error("Error no recommendation available, exception: " + error)
        return (
            jsonify({"status": "error", "strategy": strategy, "recommendation": recommendation,
                    "app_name": app_name, "app_version": app_version}),
            404,
        )

    else:
        # Return success response with recommendation data and 200 OK
        return (
            jsonify({"status": "success", "strategy": strategy, "recommendation": recommendation,
                    "app_name": app_name, "app_version": app_version}),
            200,
        )

# Error handling for 400 Bad Request
@app.errorhandler(400)
def bad_request_error(e):
    # Log the error details using the provided logger
    logger.error("400 Bad Request: " + str(e),
                 extra={
                     "info": "Bad request",
                 })
    # Return a JSON response with the error details
    return (
        jsonify({"status": "error",
                 "message": "Bad Request: " + str(e)}),
        400,
    )



# add 500 error handling:

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=80)
