user_id = "user_id"
bike_id = "bike_id"


item_features = [
    "family_id",
    "rider_height_min",
    "rider_height_max",
    "price",
]

user_features = []

features = [user_id] + [bike_id] + user_features + item_features


implicit_feedback = {
    "product_viewed": 0,
    "product_added": 0,
    "choose_service": 0,
    "add_discount": 0,
    "payment_info_entered": 0,  # booking_bike_id
    "choose_shipping_method": 0,
    "add_to_favorite": 0,
    "ask_question": 0,
    "checkout_step_completed": 0,
    "comment_show_original": 0,
    "counter_offer": 0,
    "delete_from_favourites": -5,
    "order_completed": 50,  # booking_bike_id
    "recom_bike_view": 0,  # recommended_bike_id
    "request_leasing": 0,
    "share_bike": 0,
}

test_query = """
SELECT * from "main"."rudder"."product_viewed" LIMIT 100;
"""
query = """

select user_id,
bike_id,
min(price) as price, -- min to avoid duplicates
min(family_id) as family_id,
min(rider_height_min) as rider_height_min,
min(rider_height_max) as rider_height_max,
sum(feedback) as feedback
from(

WITH user_mapping AS (
    SELECT DISTINCT anonymous_id, user_id FROM product_viewed WHERE user_id IS NOT NULL
    UNION
    SELECT DISTINCT anonymous_id, user_id FROM product_added WHERE user_id IS NOT NULL
    UNION
    SELECT DISTINCT anonymous_id, user_id FROM choose_service WHERE user_id IS NOT NULL
    UNION
    SELECT DISTINCT anonymous_id, user_id FROM add_discount WHERE user_id IS NOT NULL
    UNION
    SELECT DISTINCT anonymous_id, user_id FROM payment_info_entered WHERE user_id IS NOT NULL
    UNION
    SELECT DISTINCT anonymous_id, user_id FROM choose_shipping_method WHERE user_id IS NOT NULL
    UNION
    SELECT DISTINCT anonymous_id, user_id FROM add_to_favorite WHERE user_id IS NOT NULL
    UNION
    SELECT DISTINCT anonymous_id, user_id FROM ask_question WHERE user_id IS NOT NULL
    UNION
    SELECT DISTINCT anonymous_id, user_id FROM checkout_step_completed WHERE user_id IS NOT NULL
    UNION
    SELECT DISTINCT anonymous_id, user_id FROM comment_show_original WHERE user_id IS NOT NULL
    UNION
    SELECT DISTINCT anonymous_id, user_id FROM counter_offer WHERE user_id IS NOT NULL
    UNION
    SELECT DISTINCT anonymous_id, user_id FROM delete_from_favourites WHERE user_id IS NOT NULL
    UNION
    SELECT DISTINCT anonymous_id, user_id FROM order_completed WHERE user_id IS NOT NULL
    UNION
    SELECT DISTINCT anonymous_id, user_id FROM recom_bike_view WHERE user_id IS NOT NULL
    UNION
    SELECT DISTINCT anonymous_id, user_id FROM request_leasing WHERE user_id IS NOT NULL
    UNION
    SELECT DISTINCT anonymous_id, user_id FROM share_bike WHERE user_id IS NOT NULL
)
SELECT
    COALESCE(user_mapping.user_id, implicit_feedback.anonymous_id) AS user_id, -- Use COALESCE to fill in user_id
    implicit_feedback.anonymous_id,
    implicit_feedback.bike_id,
    bikes.price,
    bikes.family_id,
    bike_additional_infos.rider_height_min,
    bike_additional_infos.rider_height_max,
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
                WHEN event_type = 'recom_bike_view' THEN 3
                WHEN event_type = 'request_leasing' THEN 20
                WHEN event_type = 'share_bike' THEN 10
                ELSE 0
            END *
            CASE
                WHEN timestamp >= CURRENT_DATE - INTERVAL '1 day' THEN 20
                WHEN timestamp >= CURRENT_DATE - INTERVAL '3 days' THEN 15 -- This applies to 2-3 days old
                WHEN timestamp >= CURRENT_DATE - INTERVAL '7 days' THEN 12 -- This applies to 4-7 days old
                WHEN timestamp >= CURRENT_DATE - INTERVAL '14 days' THEN 9 -- This applies to 8-14 days old
                WHEN timestamp >= CURRENT_DATE - INTERVAL '21 days' THEN 6 -- This applies to 15-21 days old
                WHEN timestamp >= CURRENT_DATE - INTERVAL '1 month' THEN 4 -- This applies to 22 days to 1 month old
                WHEN timestamp >= CURRENT_DATE - INTERVAL '2 months' THEN 2 -- This applies to 1 month to 2 months old
                ELSE 1
            END) AS feedback,
        COUNT(*) OVER (PARTITION BY anonymous_id) AS anonymous_id_cnt
    FROM (
        SELECT 'product_viewed' AS event_type, anonymous_id, bike_id, timestamp FROM product_viewed WHERE timestamp >= CURRENT_DATE - INTERVAL '4 months'
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
    ) AS subquery
    GROUP BY anonymous_id, bike_id HAVING feedback > 1
) AS implicit_feedback
LEFT JOIN user_mapping ON implicit_feedback.anonymous_id = user_mapping.anonymous_id -- Join with the mapping
JOIN BUYCYCLE.PUBLIC.BIKES ON implicit_feedback.bike_id = BIKES.id
JOIN BUYCYCLE.PUBLIC.BIKE_ADDITIONAL_INFOS ON implicit_feedback.bike_id = BIKE_ADDITIONAL_INFOS.BIKE_ID
WHERE implicit_feedback.anonymous_id_cnt >= 3 AND implicit_feedback.anonymous_id_cnt <= 2000)

group by user_id, bike_id

"""
