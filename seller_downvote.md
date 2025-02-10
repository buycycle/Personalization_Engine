# response rate of seller
at least conversions with 3 differnt buyers

WITH all_sellers AS (
  SELECT
    source.seller_id AS seller_id,
    SUM(CASE WHEN source.response_status = 'seller_not_responded' THEN 0 ELSE 1 END) * 100.0 / COUNT(*) AS response_rate,
    CASE
      WHEN SUM(CASE WHEN source.response_status = 'seller_not_responded' THEN 0 ELSE 1 END) * 100.0 / COUNT(*) >= 90 THEN '90-100%'
      WHEN SUM(CASE WHEN source.response_status = 'seller_not_responded' THEN 0 ELSE 1 END) * 100.0 / COUNT(*) >= 80 THEN '80-90%'
      WHEN SUM(CASE WHEN source.response_status = 'seller_not_responded' THEN 0 ELSE 1 END) * 100.0 / COUNT(*) >= 70 THEN '70-80%'
      WHEN SUM(CASE WHEN source.response_status = 'seller_not_responded' THEN 0 ELSE 1 END) * 100.0 / COUNT(*) >= 60 THEN '60-70%'
      WHEN SUM(CASE WHEN source.response_status = 'seller_not_responded' THEN 0 ELSE 1 END) * 100.0 / COUNT(*) >= 50 THEN '50-60%'
      WHEN SUM(CASE WHEN source.response_status = 'seller_not_responded' THEN 0 ELSE 1 END) * 100.0 / COUNT(*) >= 40 THEN '40-50%'
      WHEN SUM(CASE WHEN source.response_status = 'seller_not_responded' THEN 0 ELSE 1 END) * 100.0 / COUNT(*) >= 30 THEN '30-40%'
      WHEN SUM(CASE WHEN source.response_status = 'seller_not_responded' THEN 0 ELSE 1 END) * 100.0 / COUNT(*) >= 20 THEN '20-30%'
      WHEN SUM(CASE WHEN source.response_status = 'seller_not_responded' THEN 0 ELSE 1 END) * 100.0 / COUNT(*) >= 10 THEN '10-20%'
      ELSE '0-10%'
    END AS response_rate_bin
  FROM
    (
      SELECT
        c.id,
        c.bike_id,
        c.buyer_id,
        c.seller_id,
        u.type AS seller_type,
        u.email AS seller_email,
        c.created_at,
        c.seller_first_response,
        c.seller_last_response,
        c.buyer_last_response,
        c.is_offer_start AS conv_was_started_with_offer,
        DATEDIFF(c.seller_first_response, c.created_at) AS response_time_seller_days,
        TIMESTAMPDIFF(HOUR, c.created_at, c.seller_first_response) AS response_time_seller_hours,
        bo.status AS booking_status,
        b.status AS bike_status,
        CASE
          WHEN c.seller_first_response IS NULL THEN 'seller_not_responded'
          WHEN TIMESTAMPDIFF(HOUR, c.created_at, c.seller_first_response) < 7 THEN 'seller_responded_<7hrs'
          ELSE 'seller_responded'
        END AS response_status
      FROM
        conversations c
        LEFT JOIN users u ON u.id = c.seller_id
        LEFT JOIN bikes b ON c.bike_id = b.id
        LEFT JOIN bookings bo ON c.bike_id = bo.bike_id
        AND bo.user_id = c.buyer_id
        AND bo.status NOT IN ('new', 'pending', 'failed')
      WHERE
        c.created_at > '2024-03-22'
    ) AS source
  WHERE
    source.seller_type IS NOT NULL
  GROUP BY
    source.seller_id
),
filtered_sellers AS (
  SELECT
    seller_id,
    response_rate_bin
  FROM
    all_sellers
  WHERE
    seller_id IN (
      SELECT
        seller_id
      FROM
        (
          SELECT
            seller_id,
            COUNT(DISTINCT buyer_id) AS unique_buyers
          FROM
            (
              SELECT
                c.seller_id,
                c.buyer_id
              FROM
                conversations c
              WHERE
                c.created_at > '2024-03-22'
            ) AS subquery
          GROUP BY
            seller_id
        ) AS buyer_counts
      WHERE
        unique_buyers >= 3
    )
)
SELECT
  response_rate_bin,
  COUNT(*) AS seller_count,
  COUNT(*) * 100.0 / (SELECT COUNT(*) FROM all_sellers) AS percentage_of_sellers
FROM
  filtered_sellers
GROUP BY
  response_rate_bin
ORDER BY
  response_rate_bin;


# not responding to realistic offer


### Query 1: Count of unique sellers who have at least 3 qualified offers
```sql
-- Query to count unique sellers who have at least 3 qualified offers
-- A qualified offer is defined as:
-- 1. An offer message sent by the buyer.
-- 2. The offer price is at least 80% of the listed price.
-- 3. The bike associated with the offer was not eventually sold (no approved conversation for the same bike).
WITH qualified_offers AS (
    -- Subquery to filter offers that meet the specified conditions
    SELECT
        conversations.seller_id
    FROM
        buycycle_log.sendbird_message_logs sendbird
    LEFT JOIN
        conversations ON sendbird.conversation_id = conversations.id
    WHERE
        sendbird.message_type = 'offer'  -- Only consider messages of type 'offer'
        AND sendbird.current_buyer_offer IS NOT NULL  -- Ensure the offer is not null
        AND sendbird.sent_by = conversations.buyer_id  -- Offer must be sent by the buyer
        AND sendbird.current_buyer_offer / conversations.price >= 0.8  -- Offer price must be at least 80% of the listed price
        AND NOT EXISTS (
            -- Ensure the bike was not eventually sold
            SELECT 1
            FROM conversations c2
            WHERE c2.bike_id = conversations.bike_id
            AND c2.is_approved = 1
        )
    GROUP BY
        conversations.seller_id, sendbird.sent_by, sendbird.current_buyer_offer, conversations.price, conversations.bike_id
)
SELECT
    COUNT(DISTINCT seller_id) AS distinct_seller_count  -- Count distinct sellers with qualified offers
FROM (
    -- Subquery to group sellers and filter those with at least 3 qualified offers
    SELECT
        seller_id
    FROM
        qualified_offers
    GROUP BY
        seller_id
    HAVING
        COUNT(*) >= 3  -- Only include sellers with 3 or more qualified offers
) AS sellers_with_qualified_offers;
```
### Query 2: Total Count of Unique Sellers
```sql
-- Query to count all unique sellers who have made an offer
select
    count(distinct conversations.seller_id) as total_seller_count
from
    buycycle_log.sendbird_message_logs sendbird
left join
    conversations on sendbird.conversation_id = conversations.id
where
    sendbird.message_type = 'offer'
    and sendbird.current_buyer_offer is not null;
```
### Explanation:
- **Query 1 (`qualified_seller_count`)**: This query calculates the number of unique sellers who meet the specified conditions: the bike was not sold, the message was sent by the buyer, the offer price ratio is at least 0.8, and there are at least 3 unique sellers.
- **Query 2 (`total_seller_count`)**: This query calculates the total number of unique sellers who have made an offer, without any additional conditions.
These queries are designed to be run separately, and you can use their results to perform further calculations or analysis as needed. Adjust the logic and column names to fit your actual database schema and requirements.

