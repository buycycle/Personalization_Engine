url="https://dev.recommendation.buycycle.com/recommendation"
data='{"bike_id": 14394, "user_id": 123, "distinct_id": "8130798c-db73-41f1-8d29-4b14079174a2", "n": 8, "family_id": 12}'
# Use seq to generate a list of numbers from 1 to 1000, then pipe it to xargs
seq 1000 | xargs -I % -P 1000 sh -c "curl -i -X POST '$url' -H 'Content-Type: application/json' -d '$data'"

