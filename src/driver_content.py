import pandas as pd

# features for which to prefilter for the specific recommendations.
# currently only once feature is supported, check .all or .any for supporting multiple features
# however not sure if that makes sense, maybe .any for family and family_model

# available for filtering at each request
preference_features = ["continent_id", "motor"]
# for content based
prefilter_features = ["family_id", "bike_type"]

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
]
numerical_features = ["price", "frame_size_code", "year"]

# features to overweight
numerical_features_to_overweight = ["price", "frame_size_code"]
numerical_features_overweight_factor = 4
categorical_features_to_overweight = ["bike_component_id", "bike_category_id", "bike_type"]
categorical_features_overweight_factor = 8

# main query, needs to include at least the id and the features defined above

main_query = """SELECT bikes.id as id,

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
                       family_id

                FROM bikes
                join bike_additional_infos on bikes.id = bike_additional_infos.bike_id
                left join bike_template_additional_infos on bikes.bike_template_id = bike_template_additional_infos.id
                left join countries on bikes.country_id = countries.id


                -- for non active bikes we set a one year cap for updated_at
                WHERE
                    (status = 'active') OR
                    (status NOT IN ('new', 'deleted', 'deleted_by_admin') AND TIMESTAMPDIFF(MONTH, bikes.updated_at, NOW()) < 2)

             """

main_query_dtype = {
    "id": pd.Int64Dtype(),
    "status": pd.StringDtype(),
    "bike_type": pd.Int64Dtype(),
    "bike_category_id": pd.Int64Dtype(),
    "motor": pd.Int64Dtype(),
    "continent_id": pd.Int64Dtype(),
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
}

# Define the quality features and their filter conditions
quality_features = [
    ("index", lambda df: df.index.isin(preference_mask)),
    ("bike_type", lambda df: df["bike_type"] == bike_type),
    ("price", lambda df: (df["price"] >= price * 0.8) & (df["price"] <= price * 1.2)),
    ("frame_size_code", lambda df: df["frame_size_code"] == frame_size_code),
    ("family_id", lambda df: df["family_id"] == family_id)
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
    bike_additional_infos.rider_height_min as rider_height_min,
    bike_additional_infos.rider_height_max as rider_height_max
FROM
    bikes
    join quality_scores on bikes.id = quality_scores.bike_id
    join bike_categories on bikes.bike_category_id = bike_categories.id
    join bike_additional_infos on bikes.id = bike_additional_infos.bike_id
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
    "rider_height_min": pd.Int64Dtype(),
    "rider_height_max": pd.Int64Dtype(),
}
