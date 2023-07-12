# Serverless Semantic Search for Your Website

The project demonstrates how to build a full-fledged semantic search for your web-site using Mighty, Cohere or OpenAI embeddings, [Qdrant](https://qdrant.tech/) vector search engine and [AWS Lambda](https://aws.amazon.com/lambda/).

Related repositories:

* [page-search-js](https://github.com/qdrant/page-search-js) - frontend plugin library for semantic search
* [page-search](https://github.com/qdrant/page-search) - self-hosted version of the backend and crawler in python

## How it works

AWS Lambda provides a serverless environment for running code on demand.
In the current example, we use AWS Lambda to run coordinator function that extracts embeddings from the service of your choice and constructs a search request to Qdrant.

AWS Lambda allows to keep connection to Qdrant and embedding services open between invocations, so that the latency of the search is minimal.
It is also written in Rust, so it is very fast and have minimal memory footprint, which makes it a perfect fit for serverless environment.

## How to deploy

You would need to have the following accounts (all of them have a free tier):

* [AWS](https://aws.amazon.com) for AWS Lambda. Make sure you have [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html) installed and configured.
* Either of the following embedding services (or any other of your choice, with minimal changes):
    * [Cohere](https://cohere.com)
    * [Mighty](https://max.io)
    * [OpenAI](https://openai.com)
* [Qdrant](https://qdrant.tech) for vector search.

### Prepare vector index collection

Follow instructions of how to crawl and index your web-site in [page-search](https://github.com/qdrant/page-search) repository. For indexing you can also use the `examples/setup_collections.rs` via `cargo run --release --example setup_collections` (expects the data in ``../page-search/data/articles.jsonl`)

### Configure AWS Lambda

* Create account at [AWS](https://aws.amazon.com)

### Configure ENV variables

Create a file `keys.sh` in the root of the project and fill it with the following variables:

```bash
#!/usr/bin/env bash

# Qdrant and Cohere access parameters
export QDRANT_URI="<url-to-qdrant>"
export QDRANT_API_KEY="<qdrant-api-key>"
export COHERE_API_KEY="<your API key>"

# Lambda parameters
export LAMBDA_ROLE=arn:aws:iam::XXXXXXXXXXXX:role/service-role/page-search-role
export LAMBDA_FUNCTION_NAME=page-search
export LAMBDA_REGION=us-east-1 # Where you have most of your traffic, multi AZ will be supported in the future
```

### Deploy

Run the following command to deploy the lambda function:

```bash
bash -x deploy_lambda.sh
```

You will get the following output:

```text
.......
.......

{
    "FunctionUrl": "https://hzgvs2nrxhiyuvn44adktrgre40ggykp.lambda-url.us-east-1.on.aws/", <-- example url
    "FunctionArn": "arn:aws:lambda:us-east-1:XXXXXXXXXXXX:function:qdrant-page-search",
    "AuthType": "NONE",
    "Cors": {
        "AllowHeaders": [
            "*"
        ],
        "AllowMethods": [
            "*"
        ],
        "AllowOrigins": [
            "*"
        ]
    },
    "CreationTime": "2022-11-06T13:34:08.830860Z"
}
```

Your lambda function is now deployed and ready to use from `https://hzgvs2nrxhiyuvn44adktrgre40ggykp.lambda-url.us-east-1.on.aws/`.

Test it by running the following command:

```bash
curl -X GET 'https://hzgvs2nrxhiyuvn44adktrgre40ggykp.lambda-url.us-east-1.on.aws/?q=your+query'
```

Use this link in your frontend plugin library [page-search-js](https://github.com/qdrant/page-search-js).

If you update the code, you can run:

```bash
bash -x update_lambda.sh
```

This will re-upload the code. Finally if you change any of the variables in `keys.sh`, run:

```bash
bash -x update_config.sh
```
