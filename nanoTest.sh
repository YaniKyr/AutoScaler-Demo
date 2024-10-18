#!/bin/bash

while true; do
  # Send a curl request and extract the title from the HTML response
  curl -s http://localhost:9080/productpage | grep -oP '(?<=<title>).*?(?=</title>)'
  echo ""  # Add a newline for better readability
  sleep 0.1  # Wait for 0.1 seconds before the next request
done
