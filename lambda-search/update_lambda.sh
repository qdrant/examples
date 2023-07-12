#!/usr/bin/env bash

source keys.sh

cargo lambda build --release --bin lambda --arm64 --output-format zip

# Deploy to AWS Lambda
aws lambda update-function-code --function-name $LAMBDA_FUNCTION_NAME \
  --zip-file fileb://./target/lambda/page-search/bootstrap.zip \
  --region $LAMBDA_REGION
