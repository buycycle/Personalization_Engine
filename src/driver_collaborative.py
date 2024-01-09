user_id = "anonymous_id"
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
    "product_viewed": 1,
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
    "recom_bike_view": 3,  # recommended_bike_id
    "request_leasing": 20,
    "share_bike": 10,
}

test_query = """
SELECT * from "main"."rudder"."product_viewed" LIMIT 100;
"""
query = """
SELECT implicit_feedback.anonymous_id,
    implicit_feedback.bike_id,
    bikes.price,
    bikes.family_id,
    bike_additional_infos.rider_height_min,
    bike_additional_infos.rider_height_max,
    implicit_feedback.feedback,
    implicit_feedback.anonymous_id_cnt

     FROM (
    (SELECT * FROM(
        SELECT
            anonymous_id,
            bike_id,
            SUM(
                CASE
                -- implicit feedback valuation
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
                -- implicit feedback time decay
                    WHEN timestamp >= CURRENT_DATE - INTERVAL '1 day' THEN 2
                    WHEN timestamp >= CURRENT_DATE - INTERVAL '7 days' THEN 1.5
                    ELSE 1
                END) AS feedback,
            COUNT(*) OVER (PARTITION BY anonymous_id) AS anonymous_id_cnt
        FROM (
            SELECT 'product_viewed' AS event_type, anonymous_id, bike_id, timestamp FROM product_viewed
            UNION ALL
            SELECT 'product_added', anonymous_id, bike_id, timestamp FROM product_added
            UNION ALL
            SELECT 'choose_service', anonymous_id, bike_id, timestamp FROM choose_service
            UNION ALL
            SELECT 'add_discount', anonymous_id, bike_id, timestamp FROM add_discount
            UNION ALL
            SELECT 'payment_info_entered', anonymous_id, booking_bike_id AS bike_id, timestamp FROM payment_info_entered
            UNION ALL
            SELECT 'choose_shipping_method', anonymous_id, bike_id, timestamp FROM choose_shipping_method
            UNION ALL
            SELECT 'add_to_favorite', anonymous_id, bike_id, timestamp FROM add_to_favorite
            UNION ALL
            SELECT 'ask_question', anonymous_id, bike_id, timestamp FROM ask_question
            UNION ALL
            SELECT 'checkout_step_completed', anonymous_id, bike_id, timestamp FROM checkout_step_completed
            UNION ALL
            SELECT 'comment_show_original', anonymous_id, bike_id, timestamp FROM comment_show_original
            UNION ALL
            SELECT 'counter_offer', anonymous_id, bike_id, timestamp FROM counter_offer
            UNION ALL
            SELECT 'delete_from_favourites', anonymous_id, bike_id, timestamp FROM delete_from_favourites
            UNION ALL
            SELECT 'order_completed', anonymous_id, booking_bike_id AS bike_id, timestamp FROM order_completed
            UNION ALL
            SELECT 'recom_bike_view', anonymous_id, recommended_bike_id AS bike_id, timestamp FROM recom_bike_view
            UNION ALL
            SELECT 'request_leasing', anonymous_id, bike_id, timestamp FROM request_leasing
            UNION ALL
            SELECT 'share_bike', anonymous_id, bike_id, timestamp FROM share_bike
        ) AS subquery
        GROUP BY anonymous_id, bike_id
        ORDER BY anonymous_id) AS subquery1
    -- minimal interaction limit, at least interacted with 3 bikes,
    -- maximum limit to avoid scrapers
    HAVING anonymous_id_cnt >= 3 AND anonymous_id_cnt <= 2000) as implicit_feedback)


join BUYCYCLE.PUBLIC.BIKES on implicit_feedback.bike_id = BIKES.id
join BUYCYCLE.PUBLIC.BIKE_ADDITIONAL_INFOS on implicit_feedback.bike_id = BIKE_ADDITIONAL_INFOS.BIKE_ID

"""
