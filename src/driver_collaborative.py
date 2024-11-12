user_id = "user_id"
bike_id = "bike_id"


item_features = [
    "family_id",
    "bike_category_id",
    "rider_height_min",
    "rider_height_max",
    "price",
]

user_features = []

features = [user_id] + [bike_id] + user_features + item_features


implicit_feedback = {
    "product_viewed": 0.5,
    "product_added": 2,
    "choose_service": 5,
    "add_discount": 10,
    "payment_info_entered": 20,  # booking_bike_id
    "choose_shipping_method": 20,
    "add_to_favorite": 10,
    "ask_question": 10,
    "checkout_step_completed": 12,
    "comment_show_original": 3,
    "counter_offer": 10,
    "delete_from_favourites": -5,
    "order_completed": 50,  # booking_bike_id
    "recom_bike_view": 2,  # recommended_bike_id
    "request_leasing": 20,
    "share_bike": 10,
    "show_recommendation": 0.01,
    "engaged_bike": 2,
}

test_query = """
SELECT * from "main"."rudder"."product_viewed" LIMIT 100;
"""
query = """

WITH feedback_query AS (select user_id,
bike_id,
min(price) as price, -- min to avoid duplicates
min(bike_type_id) as bike_type_id,
min(bike_category_id) as bike_category_id,
min(family_id) as family_id,
min(rider_height_min) as rider_height_min,
min(rider_height_max) as rider_height_max,
sum(feedback) as feedback
from(

--- backfill user_id from all interactions
WITH user_mapping AS (
    SELECT DISTINCT anonymous_id, COALESCE(user_id, anonymous_id) AS user_id FROM product_viewed
    UNION
    SELECT DISTINCT anonymous_id, COALESCE(user_id, anonymous_id) FROM product_added
    UNION
    SELECT DISTINCT anonymous_id, COALESCE(user_id, anonymous_id) FROM choose_service
    UNION
    SELECT DISTINCT anonymous_id, COALESCE(user_id, anonymous_id) FROM add_discount
    UNION
    SELECT DISTINCT anonymous_id, COALESCE(user_id, anonymous_id) FROM payment_info_entered
    UNION
    SELECT DISTINCT anonymous_id, COALESCE(user_id, anonymous_id) FROM choose_shipping_method
    UNION
    SELECT DISTINCT anonymous_id, COALESCE(user_id, anonymous_id) FROM add_to_favorite
    UNION
    SELECT DISTINCT anonymous_id, COALESCE(user_id, anonymous_id) FROM ask_question
    UNION
    SELECT DISTINCT anonymous_id, COALESCE(user_id, anonymous_id) FROM checkout_step_completed
    UNION
    SELECT DISTINCT anonymous_id, COALESCE(user_id, anonymous_id) FROM comment_show_original
    UNION
    SELECT DISTINCT anonymous_id, COALESCE(user_id, anonymous_id) FROM counter_offer
    UNION
    SELECT DISTINCT anonymous_id, COALESCE(user_id, anonymous_id) FROM delete_from_favourites
    UNION
    SELECT DISTINCT anonymous_id, COALESCE(user_id, anonymous_id) FROM order_completed
    UNION
    SELECT DISTINCT anonymous_id, COALESCE(user_id, anonymous_id) FROM recom_bike_view
    UNION
    SELECT DISTINCT anonymous_id, COALESCE(user_id, anonymous_id) FROM request_leasing
    UNION
    SELECT DISTINCT anonymous_id, COALESCE(user_id, anonymous_id) FROM share_bike
    UNION
    SELECT DISTINCT anonymous_id, COALESCE(user_id, anonymous_id) FROM show_recommendations
    UNION
    SELECT DISTINCT anonymous_id, COALESCE(user_id, anonymous_id) FROM engaged_bike_view
)
SELECT
    user_mapping.user_id,
    implicit_feedback.anonymous_id,
    implicit_feedback.bike_id,
    FLOOR(bikes.price / 200) * 200 price,
    bikes.bike_type_id,
    bikes.bike_category_id,
    bikes.family_id,
    FLOOR(bike_additional_infos.rider_height_min / 10) * 10 AS rider_height_min,
    FLOOR(bike_additional_infos.rider_height_max / 10) * 10 AS rider_height_max,
    implicit_feedback.feedback,
    implicit_feedback.anonymous_id_cnt
FROM (
    SELECT
        anonymous_id,
        bike_id,
        SUM(
            CASE
                WHEN event_type = 'product_viewed' THEN 1
                WHEN event_type = 'product_added' THEN 2
                WHEN event_type = 'choose_service' THEN 5
                WHEN event_type = 'add_discount' THEN 10
                WHEN event_type = 'payment_info_entered' THEN 20
                WHEN event_type = 'choose_shipping_method' THEN 20
                WHEN event_type = 'add_to_favorite' THEN 10
                WHEN event_type = 'ask_question' THEN 10
                WHEN event_type = 'checkout_step_completed' THEN 12
                WHEN event_type = 'comment_show_original' THEN 3
                WHEN event_type = 'counter_offer' THEN 10
                WHEN event_type = 'delete_from_favourites' THEN -5
                WHEN event_type = 'order_completed' THEN 50
                WHEN event_type = 'recom_bike_view' THEN 5
                WHEN event_type = 'request_leasing' THEN 20
                WHEN event_type = 'share_bike' THEN 10
                WHEN event_type = 'show_recommendation' THEN 0
                WHEN event_type = 'engaged_bike' THEN 1
                ELSE 0
            END *
            CASE
            -- phase out multiplication factor over time
                WHEN timestamp >= CURRENT_DATE - INTERVAL '1 day' THEN 2
                WHEN timestamp >= CURRENT_DATE - INTERVAL '3 days' THEN 1.5 -- This applies to 2-3 days old
                WHEN timestamp >= CURRENT_DATE - INTERVAL '7 days' THEN 1.25 -- This applies to 4-7 days old
                WHEN timestamp >= CURRENT_DATE - INTERVAL '14 days' THEN 1 -- This applies to 8-14 days old
                WHEN timestamp >= CURRENT_DATE - INTERVAL '21 days' THEN 0.75 -- This applies to 15-21 days old
                WHEN timestamp >= CURRENT_DATE - INTERVAL '1 month' THEN 0.5 -- This applies to 22 days to 1 month old
                ELSE 0.25
            END) AS feedback,
        COUNT(*) OVER (PARTITION BY anonymous_id) AS anonymous_id_cnt
    --- restrict the data we look at for product viewed and recom_bike_viewed to 4 month, rest 12 month
    FROM (
        SELECT 'product_viewed' AS event_type, anonymous_id, bike_id, timestamp FROM product_viewed WHERE timestamp >= CURRENT_DATE - INTERVAL '3 months'
        UNION ALL
        SELECT 'product_added', anonymous_id, bike_id, timestamp FROM product_added WHERE timestamp >= CURRENT_DATE - INTERVAL '12 months'
        UNION ALL
        SELECT 'choose_service', anonymous_id, bike_id, timestamp FROM choose_service WHERE timestamp >= CURRENT_DATE - INTERVAL '12 months'
        UNION ALL
        SELECT 'add_discount', anonymous_id, bike_id, timestamp FROM add_discount WHERE timestamp >= CURRENT_DATE - INTERVAL '12 months'
        UNION ALL
        SELECT 'payment_info_entered', anonymous_id, booking_bike_id AS bike_id, timestamp FROM payment_info_entered WHERE timestamp >= CURRENT_DATE - INTERVAL '12 months'
        UNION ALL
        SELECT 'choose_shipping_method', anonymous_id, bike_id, timestamp FROM choose_shipping_method WHERE timestamp >= CURRENT_DATE - INTERVAL '12 months'
        UNION ALL
        SELECT 'add_to_favorite', anonymous_id, bike_id, timestamp FROM add_to_favorite WHERE timestamp >= CURRENT_DATE - INTERVAL '12 months'
        UNION ALL
        SELECT 'ask_question', anonymous_id, bike_id, timestamp FROM ask_question WHERE timestamp >= CURRENT_DATE - INTERVAL '12 months'
        UNION ALL
        SELECT 'checkout_step_completed', anonymous_id, bike_id, timestamp FROM checkout_step_completed WHERE timestamp >= CURRENT_DATE - INTERVAL '12 months'
        UNION ALL
        SELECT 'comment_show_original', anonymous_id, bike_id, timestamp FROM comment_show_original WHERE timestamp >= CURRENT_DATE - INTERVAL '12 months'
        UNION ALL
        SELECT 'counter_offer', anonymous_id, bike_id, timestamp FROM counter_offer WHERE timestamp >= CURRENT_DATE - INTERVAL '12 months'
        UNION ALL
        SELECT 'delete_from_favourites', anonymous_id, bike_id, timestamp FROM delete_from_favourites WHERE timestamp >= CURRENT_DATE - INTERVAL '12 months'
        UNION ALL
        SELECT 'order_completed', anonymous_id, booking_bike_id AS bike_id, timestamp FROM order_completed WHERE timestamp >= CURRENT_DATE - INTERVAL '12 months'
        UNION ALL
        SELECT 'recom_bike_view', anonymous_id, recommended_bike_id AS bike_id, timestamp FROM recom_bike_view WHERE timestamp >= CURRENT_DATE - INTERVAL '4 months'
        UNION ALL
        SELECT 'request_leasing', anonymous_id, bike_id, timestamp FROM request_leasing WHERE timestamp >= CURRENT_DATE - INTERVAL '12 months'
        UNION ALL
        SELECT 'share_bike', anonymous_id, bike_id, timestamp FROM share_bike WHERE timestamp >= CURRENT_DATE - INTERVAL '12 months'
        UNION ALL
        SELECT 'show_recommendation' AS event_type, anonymous_id, CAST(value AS INTEGER) AS bike_id, timestamp FROM show_recommendations, LATERAL FLATTEN(input => SPLIT(recommendation_product_ids, ',')) WHERE timestamp >= CURRENT_DATE - INTERVAL '1 months'
        UNION ALL
        SELECT 'engaged_bike', anonymous_id, bike_id, timestamp FROM engaged_bike_view WHERE timestamp >= CURRENT_DATE - INTERVAL '3 months'
    ) AS subquery
    GROUP BY anonymous_id, bike_id HAVING feedback > 6
) AS implicit_feedback
LEFT JOIN user_mapping ON implicit_feedback.anonymous_id = user_mapping.anonymous_id -- Join with the mapping
JOIN BUYCYCLE.PUBLIC.BIKES ON implicit_feedback.bike_id = BIKES.id
JOIN BUYCYCLE.PUBLIC.BIKE_ADDITIONAL_INFOS ON implicit_feedback.bike_id = BIKE_ADDITIONAL_INFOS.BIKE_ID
--- user must have interacted with at least 3 and max 4000 bikes
WHERE implicit_feedback.anonymous_id_cnt >= 3 AND implicit_feedback.anonymous_id_cnt <= 4000)

group by user_id, bike_id),

--- preference table, extract budget_min and _max as well as bike_category_id from stated preference per user
user_preference AS (
    SELECT
      users.id as user_id,
      BUYCYCLE.PUBLIC.bike_categories.id as bike_category_id,
        CASE
        WHEN budget IS NULL THEN NULL
        WHEN budget LIKE 'More than%' THEN CAST(REPLACE(REPLACE(budget, '€', ''), 'More than ', '') AS DECIMAL)
        ELSE CAST(REPLACE(REPLACE(SUBSTRING(budget, 1, CHARINDEX('-', budget) - 1), '€', ''), '.', '') AS DECIMAL)
      END AS budget_min,
      CASE
        WHEN budget IS NULL THEN NULL
        WHEN budget LIKE 'More than%' THEN 20000
        ELSE CAST(REPLACE(REPLACE(SUBSTRING(budget, CHARINDEX('-', budget) + 1, LEN(budget)), '€', ''), '.', '') AS DECIMAL)
      END AS budget_max
    FROM users
    left join BUYCYCLE.PUBLIC.bike_categories ON context_traits_what_kind_of_rider_style = bike_categories.name

)
--- join feedback with preference table
SELECT
    fq.user_id,
    fq.bike_id,
    min(fq.price) as price,
    min(fq.bike_type_id) as bike_type_id,
    min(fq.bike_category_id) as bike_category_id,
    min(fq.family_id) as family_id,
    min(fq.rider_height_min) as rider_height_min,
    min(fq.rider_height_max) as rider_height_max,
    SUM(fq.feedback) as feedback

FROM feedback_query fq
GROUP BY
    fq.user_id,
    fq.bike_id,
    fq.bike_type_id,
    fq.bike_category_id
"""
