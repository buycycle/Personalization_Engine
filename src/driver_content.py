import pandas as pd

# features for which to prefilter for the specific recommendations.
# currently only once feature is supported, check .all or .any for supporting multiple features
# however not sure if that makes sense, maybe .any for family and family_model

# available for filtering at each request
preference_features = [
    "continent_id",
    "motor",
    "price",
    "frame_size_code",
    "bike_category_id",
]
# for content based
prefilter_features = ["family_id", "bike_type", "brand"]

# for the content based recommendation we disregard prefilter_features and use generic features that represent the qualities of the bike
# categorical_features and numerical_features features to consider in the generic recommendations

categorical_features = [
    "motor",
    "bike_component_id",
    "bike_category_id",
    "bike_type",
    "brake_type_code",
    "frame_material_code",
    "shifting_code",
    "color",
    "suspension",
    "continent_id",
    "brand",
    "family_id",
]
numerical_features = ["price", "frame_size_code", "year"]

# features to overweight
numerical_features_to_overweight = ["price", "frame_size_code"]
numerical_features_overweight_factor = 4
categorical_features_to_overweight = [
    "family_id",
    "bike_component_id",
    "bike_category_id",
    "bike_type",
    "brand",
]
categorical_features_overweight_factor = 20

# main query, needs to include at least the id and the features defined above

main_query = """SELECT bikes.id as id,
                       bikes.bike_template_id as template_id,
                       quality_scores.score as quality_score,

                       bikes.status as status,
                       -- categorizing
                       bike_type_id as bike_type,
                       bike_category_id,
                       motor,
                       countries.continent_id as continent_id,

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

                       COALESCE(bike_template_additional_infos.suspension, 'NULL') as suspension,

                       -- quite specific
                       family_id,
                       
                       -- brand information
                       brands.slug as brand

                FROM bikes
                join bike_additional_infos on bikes.id = bike_additional_infos.bike_id
                left join bike_template_additional_infos on bikes.bike_template_id = bike_template_additional_infos.id
                left join countries on bikes.country_id = countries.id
                join quality_scores on bikes.id = quality_scores.bike_id
                join brands on bikes.brand_id = brands.id


                -- for non active bikes we set a one year cap for updated_at
                WHERE
                    (status = 'active') OR
                    (status NOT IN ('new', 'deleted', 'deleted_by_admin', 'rejected', 'deleted_draft', 'inactive', 'blocked') AND TIMESTAMPDIFF(WEEK, bikes.updated_at, NOW()) <= 2) AND
                    (bikes.bike_template_id != 79204) AND
                    (quality_scores.score > 30)


             """

main_query_dtype = {
    "id": pd.Int64Dtype(),
    "template_id": pd.Int64Dtype(),
    "quality_score": pd.Int64Dtype(),
    "status": pd.StringDtype(),
    "bike_type": pd.Int64Dtype(),
    "bike_category_id": pd.Int64Dtype(),
    "motor": pd.Int64Dtype(),
    "continent_id": pd.Int64Dtype(),
    # frame_size_code as string, we convert it in frame_size_code_to_numeric
    "frame_size_code": pd.StringDtype(),
    "price": pd.Float64Dtype(),
    "brake_type_code": pd.StringDtype(),
    "frame_material_code": pd.StringDtype(),
    "shifting_code": pd.StringDtype(),
    "year": pd.Int64Dtype(),
    "bike_component_id": pd.Int64Dtype(),
    "color": pd.StringDtype(),
    "suspension": pd.StringDtype(),
    "family_id": pd.Int64Dtype(),
    "brand": pd.StringDtype(),
}

# not used, just to make sure what feautures are necessary
quality_features = [
    "bike_type",
    "price",
    "rider_height_max",
    "rider_height_min",
    "family_id",
    "slug",
    "is_ebike",
    "is_frameset",
    "brand",
]


quality_query = """SELECT
    bikes.id as id,
    price,
    bike_additional_infos.frame_size as frame_size_code,
    quality_scores.score as quality_score,
    family_id,
    bike_type_id as bike_type,
    bikes.slug as slug,
    bike_categories.slug as category,
    bikes.bike_category_id as bike_category_id,
    bike_additional_infos.rider_height_min as rider_height_min,
    bike_additional_infos.rider_height_max as rider_height_max,
    COALESCE(bike_template_additional_infos.is_ebike, 0) as is_ebike,
    COALESCE(bike_template_additional_infos.is_frameset, 0) as is_frameset,
    brands.slug as brand
FROM
    bikes
    join quality_scores on bikes.id = quality_scores.bike_id
    join bike_categories on bikes.bike_category_id = bike_categories.id
    join bike_additional_infos on bikes.id = bike_additional_infos.bike_id
    join bike_template_additional_infos on bikes.bike_template_id = bike_template_additional_infos.bike_template_id
    join brands on bikes.brand_id = brands.id



WHERE
    status = 'active'
GROUP BY
    id,
    price,
    frame_size_code,
    family_id,
    bike_type,
    slug,
    category,
    bike_category_id,
    rider_height_min,
    rider_height_max
ORDER BY
    quality_score DESC
"""

quality_query_dtype = {
    "id": pd.Int64Dtype(),
    "bike_type": pd.Int64Dtype(),
    "price": pd.Float64Dtype(),
    "family_id": pd.Int64Dtype(),
    # frame_size_code as string, we convert it in frame_size_code_to_numeric
    "frame_size_code": pd.StringDtype(),
    "quality_score": pd.Int64Dtype(),
    "slug": pd.StringDtype(),
    "category": pd.StringDtype(),
    "bike_category_id": pd.Int64Dtype(),
    "rider_height_min": pd.Float64Dtype(),
    "rider_height_max": pd.Float64Dtype(),
    "is_ebike": pd.Int64Dtype(),
    "is_frameset": pd.Int64Dtype(),
    "brand": pd.StringDtype(),
}


user_preference_query = """
WITH preference_table AS (
    SELECT
        USER_BIKE_PREFERENCES.USER_ID,
        COALESCE(PARSE_JSON(USER_BIKE_PREFERENCES.PREFERENCES)['max_price']::NUMBER, 20000) AS max_price,
        COALESCE(CATEGORY.VALUE::STRING, 'none') AS category,
        COALESCE(FRAME_SIZE.VALUE::STRING, 'none') AS frame_size
    FROM
        BUYCYCLE.PUBLIC.USER_BIKE_PREFERENCES,
        LATERAL FLATTEN(input => PARSE_JSON(USER_BIKE_PREFERENCES.PREFERENCES)['categories'], OUTER => TRUE) CATEGORY,
        LATERAL FLATTEN(input => PARSE_JSON(USER_BIKE_PREFERENCES.PREFERENCES)['frame_sizes'], OUTER => TRUE) FRAME_SIZE
)
SELECT user_id,
    max_price,
    COALESCE(bike_categories.id, 0) as category_id,
    frame_size
FROM preference_table
LEFT JOIN BUYCYCLE.PUBLIC.bike_categories ON LOWER(category) = LOWER(bike_categories.name)
"""
user_preference_query_dtype = {
    "user_id": pd.Int64Dtype(),
    "max_price": pd.Int64Dtype(),
    "category_id": pd.Int64Dtype(),
    "frame_size": pd.StringDtype(),
}

reranking_query = """
WITH preference_table AS (
    SELECT
        USER_BIKE_PREFERENCES.USER_ID,
        COALESCE(PARSE_JSON(USER_BIKE_PREFERENCES.PREFERENCES)['max_price']::NUMBER, 20000) AS max_price,
        COALESCE(CATEGORY.VALUE::STRING, 'none') AS category,
        COALESCE(FRAME_SIZE.VALUE::STRING, 'none') AS frame_size
    FROM
        BUYCYCLE.PUBLIC.USER_BIKE_PREFERENCES,
        LATERAL FLATTEN(input => PARSE_JSON(USER_BIKE_PREFERENCES.PREFERENCES)['categories'], OUTER => TRUE) CATEGORY,
        LATERAL FLATTEN(input => PARSE_JSON(USER_BIKE_PREFERENCES.PREFERENCES)['frame_sizes'], OUTER => TRUE) FRAME_SIZE
)
SELECT user_id,
    max_price,
    COALESCE(bike_categories.id, 0) as category_id,
    frame_size
FROM preference_table
LEFT JOIN BUYCYCLE.PUBLIC.bike_categories ON LOWER(category) = LOWER(bike_categories.name)
"""
reranking_query_dtypes = {
    "user_id": pd.Int64Dtype(),
    "max_price": pd.Int64Dtype(),
    "category_id": pd.Int64Dtype(),
    "frame_size": pd.StringDtype(),
}
