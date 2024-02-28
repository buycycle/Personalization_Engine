import pandas as pd

# features for which to prefilter for the specific recommendations.
# currently only once feature is supported, check .all or .any for supporting multiple features
# however not sure if that makes sense, maybe .any for family and family_model
# needs some thought

# prefilter_features = ["family_model_id", "family_id", "brand_id"]
prefilter_features = ["family_id"]

# for the content based recommendation we disregard prefilter_features and use generic features that represent the qualities of the bike
# categorical_features and numerical_features features to consider in the generic recommendations

categorical_features = [
    "motor",
    "bike_component_id",
    "bike_category_id",
    "bike_type_id",
    "brake_type_code",
    "frame_material_code",
    "shifting_code",
    "color",
]
numerical_features = ["price", "frame_size_code", "year"]

# features to overweight
numerical_features_to_overweight = ["price", "frame_size_code"]
numerical_features_overweight_factor = 4
categorical_features_to_overweight = ["bike_component_id", "bike_category_id", "bike_type_id"]
categorical_features_overweight_factor = 2

# main query, needs to include at least the id and the features defined above

main_query = """SELECT bikes.id as id,

                       bikes.status as status,
                       -- categorizing
                       bike_type_id,
                       bike_category_id,
                       motor,

                       -- cetegorizing fuzzy

                       bike_additional_infos.frame_size as frame_size_code,



                       -- very important
                       price,
                       brake_type_code,
                       frame_material_code,
                       shifting_code,


                       -- important

                       year,

                       bike_component_id,

                       -- find similarity between hex codes
                       color,

                       -- quite specific
                       family_id

                FROM bikes
                join bike_additional_infos on bikes.id = bike_additional_infos.bike_id


                -- for non active bikes we set a one year cap for updated_at
                WHERE
                    (status = 'active') OR
                    (status NOT IN ('new', 'deleted', 'deleted_by_admin') AND TIMESTAMPDIFF(MONTH, bikes.updated_at, NOW()) < 2)

             """

main_query_dtype = {
    "id": pd.Int64Dtype(),
    "status": pd.StringDtype(),
    "bike_type_id": pd.Int64Dtype(),
    "bike_category_id": pd.Int64Dtype(),
    "motor": pd.Int64Dtype(),
    "price": pd.Float64Dtype(),
    "year": pd.Int64Dtype(),
    "bike_component_id": pd.Int64Dtype(),
    "family_id": pd.Int64Dtype(),
    # frame_size_code as string, we convert it in frame_size_code_to_numeric
    "frame_size_code": pd.StringDtype(),
}

popularity_query = """SELECT
    bike_id AS id,
    bike_price AS price,
    bike_frame_size AS frame_size_code,
    bike_family_id AS family_id,
    bike_type_id,
    COUNT(*) AS count_of_visits
FROM
    product_viewed
WHERE
    bike_status = 'active'
GROUP BY
    bike_id,
    bike_price,
    bike_frame_size,
    bike_family_id,
    bike_type_id
HAVING
    COUNT(*) > 10
ORDER BY
    count_of_visits DESC
"""

popularity_query_dtype = {
    "id": pd.Int64Dtype(),
    "bike_type_id": pd.Int64Dtype(),
    "price": pd.Float64Dtype(),
    "family_id": pd.Int64Dtype(),
    # frame_size_code as string, we convert it in frame_size_code_to_numeric
    "frame_size_code": pd.StringDtype(),
    "count_of_visits": pd.Int64Dtype(),
}
