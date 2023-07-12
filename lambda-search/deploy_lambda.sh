#!/usr/bin/env bash

source keys.sh

cargo lambda build --release --bin lambda --arm64 --output-format zip

# Clear previous versions
aws lambda delete-function-url-config --region $LAMBDA_REGION --function-name $LAMBDA_FUNCTION_NAME
aws lambda delete-function --region $LAMBDA_REGION --function-name $LAMBDA_FUNCTION_NAME

# Deploy to AWS Lambda
aws lambda create-function --function-name $LAMBDA_FUNCTION_NAME \
  --handler bootstrap \
  --architectures arm64 \
  --zip-file fileb://./target/lambda/lambda/bootstrap.zip \
  --runtime provided.al2 \
  --region $LAMBDA_REGION \
  --role $LAMBDA_ROLE \
  --environment "Variables={QDRANT_URI=$QDRANT_URI,QDRANT_API_KEY=$QDRANT_API_KEY,COHERE_API_KEY=$COHERE_API_KEY}" \
  --tracing-config Mode=Active

# Grant public access to the function
# https://docs.amazonaws.cn/en_us/lambda/latest/dg/urls-tutorial.html
aws lambda add-permission \
    --function-name $LAMBDA_FUNCTION_NAME \
    --action lambda:InvokeFunctionUrl \
    --principal "*" \
    --function-url-auth-type "NONE" \
    --region $LAMBDA_REGION \
    --statement-id url

# Assign URL to the function
# https://docs.aws.amazon.com/de_de/cli/latest/reference/lambda/create-function-url-config.html
aws lambda create-function-url-config \
  --function-name $LAMBDA_FUNCTION_NAME \
  --region $LAMBDA_REGION \
  --cors "AllowOrigins=*,AllowMethods=*,AllowHeaders=*" \
  --auth-type NONE

