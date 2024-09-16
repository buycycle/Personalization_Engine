import sys
import time
from src.data_content import create_data_model_content
from src.driver_content import (
    main_query,
    main_query_dtype,
    quality_query,
    quality_query_dtype,
    user_preference_query,
    user_preference_query_dtype,
    categorical_features,
    numerical_features,
    preference_features,
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

# Check if the second command line argument is 'test' and set a flag accordingly
is_test_mode = len(sys.argv) > 2 and sys.argv[2] == "test"
# Add limits to the queries if in test mode
main_query_limit = " LIMIT 8000" if is_test_mode else ""
collaborative_query_limit = " LIMIT 20000" if is_test_mode else ""

create_data_model_content(
    main_query=main_query + main_query_limit,
    main_query_dtype=main_query_dtype,
    quality_query=quality_query,
    quality_query_dtype=quality_query_dtype,
    user_preference_query=user_preference_query,
    user_preference_query_dtype=user_preference_query_dtype,
    categorical_features=categorical_features,
    numerical_features=numerical_features,
    preference_features=preference_features,
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
    query=query + collaborative_query_limit,
    user_id=user_id,
    bike_id=bike_id,
    user_features=user_features,
    item_features=item_features,
    update_model=update_model,
    path=path,
)

print(f"created_data_model_collaborative with Test AUC: {test_auc}")

# sleep to make sure the data saved
time.sleep(4)
