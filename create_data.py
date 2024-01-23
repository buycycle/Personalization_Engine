import sys
import time
from src.data_content import create_data_model_content

from src.driver_content import (
    main_query,
    main_query_dtype,
    popularity_query,
    popularity_query_dtype,
    categorical_features,
    numerical_features,
    prefilter_features,
    numerical_features_to_overweight,
    numerical_features_overweight_factor,
    categorical_features_to_overweight,
    categorical_features_overweight_factor,
)

from src.driver_collaborative import user_id, bike_id, item_features, user_features, query
from src.collaborative import create_data_model_collaborative, update_model

# if there is a command line argument, use it as path, else use './data/'
path = sys.argv[1] if len(sys.argv) > 1 else "./data/"

create_data_model_content(
    main_query=main_query,
    main_query_dtype=main_query_dtype,
    popularity_query=popularity_query,
    popularity_query_dtype=popularity_query_dtype,
    categorical_features=categorical_features,
    numerical_features=numerical_features,
    prefilter_features=prefilter_features,
    numerical_features_to_overweight=numerical_features_to_overweight,
    numerical_features_overweight_factor=numerical_features_overweight_factor,
    categorical_features_to_overweight=categorical_features_to_overweight,
    categorical_features_overweight_factor=categorical_features_overweight_factor,
    status=["active"],
    metric="euclidean",
    path=path,
)

print("created_data_model_content")

test_auc = create_data_model_collaborative(
    DB="DB_EVENTS",
    driver="snowflake",
    query=query,
    user_id=user_id,
    bike_id=bike_id,
    user_features=user_features,
    item_features=item_features,
    update_model=update_model,
    path=path,
)

print(f"created_data_model_collaborative with Test AUC: {test_auc}")

# sleep for 10 seconds to make sure the data saved
time.sleep(10)
