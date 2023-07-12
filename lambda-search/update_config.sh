#!/usr/bin/env bash

source keys.sh

case $EMBEDDING in
("COHERE"|"OPEN_AI"))
  export EMBEDDING_ENV="EMBED_API_KEY=$EMBED_API_KEY"
  ;;
"MIGHTY")
  export EMBEDDING_ENV="MIGHTY_URI=$MIGHTY_URI"
  ;;
esac

aws lambda update-function-configuration \
  --function-name $LAMBDA_FUNCTION_NAME \
  --environment "Variables={QDRANT_URI=$QDRANT_URI,QDRANT_API_KEY=$QDRANT_API_KEY,EMBEDDING=$EMBEDDING,COHERE_API_KEY=$COHERE_API_KEY}"
