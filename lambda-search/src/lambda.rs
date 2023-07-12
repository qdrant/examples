use lambda_http::{run, service_fn, Body, Error, Request, RequestExt, Response};
use qdrant_client::prelude::*;
use qdrant_client::qdrant::{value::Kind, Struct};
use reqwest::Client;
use serde::Deserialize;

const SEARCH_LIMIT: u64 = 5;

fn error_response(code: u16, message: &str) -> Response<Body> {
    Response::builder()
        .status(code)
        .body(Body::from(message))
        .unwrap()
}

#[derive(Deserialize)]
struct CohereResponse {
    outputs: Vec<Vec<f32>>
}

async fn function_handler(
    event: Request,
    client: &Client,
    cohere_api_key: &str,
    qdrant_client: &QdrantClient,
    collection_name: &str,
) -> Result<Response<Body>, Error> {
    let Some(params) = event.query_string_parameters_ref() else {
        return Ok(error_response(400, "Missing query string parameters"));
    };
    let Some(query) = params.first("q") else {
        return Ok(error_response(400, "Missing query string parameter `q`"));
    };
    let CohereResponse { outputs } = client
.post("https://api.cohere.ai/embed")
        .header("Authorization", &format!("Bearer {cohere_api_key}"))
        .header("Content-Type", "application/json")
        .header("Cohere-Version", "2021-11-08")
        .body(format!("{{\"text\":[\"{query}\"],\"model\":\"small\"}}"))
        .send()
        .await?
        .json()
        .await?;

    let response_body: String = qdrant_client
        .search_points(&SearchPoints {
            collection_name: collection_name.to_string(),
            vector: outputs.into_iter().next().ok_or("Empty output from embedding")?,
            limit: SEARCH_LIMIT as u64,
            with_payload: Some(true.into()),
            ..Default::default()
        })
        .await?
        .result
        .into_iter()
        .map(|p| {
            format!(
                "{}\n",
                Value {
                    kind: Some(Kind::StructValue(Struct { fields: p.payload }))
                }
            )
        })
        .collect();

    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(response_body.into())
        .map_err(Box::new)?)
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    env_logger::init();

    let client = Client::builder().build().unwrap();
    let cohere_api_key = std::env::var("COHERE_API_KEY").expect("need COHERE_API_KEY set");
    let collection_name = std::env::var("COLLECTION_NAME").expect("need COLLECTION_NAME set");
    let qdrant_uri = std::env::var("QDRANT_URI").expect("need QDRANT_URI set");
    let mut config = QdrantClientConfig::from_url(&qdrant_uri);
    config.api_key = std::env::var("QDRANT_API_KEY").ok();
    let qdrant_client = QdrantClient::new(Some(config)).expect("Failed to connect to Qdrant");
    if !qdrant_client
        .has_collection(collection_name.clone())
        .await?
    {
        panic!("Collection {} not found", collection_name);
    }

    run(service_fn(|req| {
        function_handler(
            req,
            &client,
            &cohere_api_key,
            &qdrant_client,
            &collection_name,
        )
    }))
    .await
}
