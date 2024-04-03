#!/bin/bash
url="https://ab.recommendation.buycycle.com/recommendation"
data_template='{
  "strategy": "product_page",
  "bike_id": 21091,
  "user_id": 18742,
  "device_id": "bccd547b-6736-41a2-8543-677b724fb2a1",
  "frame_size_code":"58",
  "price": 2750,
  "family_id": 803,
  "n": 12
}'
# Total number of requests to send
total_requests=100
# Initialize an associative array to hold the count of HTTP status codes and versions
declare -A status_code_count
declare -A version_count
# Print initial information
current_date_time=$(date '+%Y-%m-%d %H:%M:%S')
echo "Date and Time: $current_date_time"
echo "Address: $url"
echo "Request Payload: $data_template"
echo "Total requests to be sent: $total_requests"
# Loop total_requests times
for i in $(seq 1 $total_requests); do
  # Execute the curl command and capture the headers and body
  response=$(curl -s -i -X POST "$url" \
    -H "Content-Type: application/json" \
    -H "strategy: Generic" \
    -H "model: price-dev" \
    -H "version: canary-001" \
    -d "$data_template")
  # Extract the HTTP status code and version from the response
  http_code=$(echo "$response" | grep HTTP | tail -1 | awk '{print $2}')
  version=$(echo "$response" | grep -i "^version:" | awk '{print $2}' | tr -d '\r')
  # Increment the count for this HTTP status code and version
  ((status_code_count[$http_code]++))
  ((version_count[$version]++))
done
# Print the count of each HTTP status code
echo "HTTP Status Code Counts:"
for code in "${!status_code_count[@]}"; do
  echo "Status Code $code: ${status_code_count[$code]}"
done
# Print the count of each version
echo "Version Counts:"
for ver in "${!version_count[@]}"; do
  echo "Version $ver: ${version_count[$ver]}"
done
# Print completion information
echo "Total requests sent: $total_requests"


