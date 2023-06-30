//! This program can be used to set up the collection in Qdrant.
//! `cargo run --release --bin setup_collection`
//! The program will output one dot per point inserted and the current number every 100 points.

use qdrant_client::prelude::*;
use qdrant_client::qdrant::{
    value::Kind, vectors_config::Config, CreateCollection, Distance, FieldType, PointId,
    PointStruct, Value, VectorParams, Vectors, VectorsConfig,
};
use reqwest::Client;
use serde::Deserialize;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

#[derive(Deserialize)]
struct OpenaiResponse {
    data: OpenaiData,
}

#[derive(Deserialize)]
struct OpenaiData {
    embedding: Vec<f32>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::builder().build()?;
    let openai_api_key = std::env::var("OPENAI_API_KEY").expect("need OPENAI_API_KEY set");
    let collection_name = std::env::var("COLLECTION_NAME").expect("needs COLLECTION_NAME set");
    let qdrant_uri = std::env::var("QDRANT_URI").expect("need QDRANT_URI set");
    let mut config = QdrantClientConfig::from_url(&qdrant_uri);
    config.api_key = std::env::var("QDRANT_API_KEY").ok();
    let qdrant_client = QdrantClient::new(Some(config)).expect("Failed to connect to Qdrant");

    if !qdrant_client.has_collection(&collection_name).await? {
        qdrant_client
            .create_collection(&CreateCollection {
                collection_name: collection_name.clone(),
                vectors_config: Some(VectorsConfig {
                    config: Some(Config::Params(VectorParams {
                        size: 1536,
                        distance: Distance::Cosine as i32,
                        ..Default::default()
                    })),
                }),
                ..Default::default()
            })
            .await?;
        for field_name in ["sections", "tag"] {
            qdrant_client
                .create_field_index(&collection_name, field_name, FieldType::Keyword, None, None)
                .await?;
        }
    }
    let file = std::env::args()
        .nth(1)
        .expect("Needs the JSONL file in the first argument");
    let mut points = Vec::new();
    let abstracts = File::open(&file).expect("couldn't open JSONL");
    let abstracts = BufReader::new(abstracts);
    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();
    let i = &mut 1;
    for line in abstracts.lines() {
        let payload: HashMap<String, Value> = serde_json::from_str(&line?)?;
        let text = payload.get("text");
        let Some(Value {
            kind: Some(Kind::StringValue(text)),
        }) = text
        else {
            panic!("text isn't a string")
        };

        let OpenaiResponse {
            data: OpenaiData { embedding },
        } = client
            .post("https://api.openai.com/v1/embedding")
            .header("Authorization", &format!("Bearer {openai_api_key}"))
            .header("Content-Type", "application/json")
            .body(format!(
                "{{\"input\":\"{text}\",\"model\":\"text-embedding-ada-002\"}}"
            ))
            .send()
            .await?
            .json()
            .await?;
        points.push(PointStruct {
            id: Some(PointId::from(std::mem::replace(i, *i + 1) as u64)),
            payload: payload.clone(),
            vectors: Some(Vectors::from(embedding)),
        });
        write!(stdout, ".")?;
        if *i % 100 == 0 {
            write!(stdout, "{}", i)?;
            qdrant_client
                .upsert_points(&collection_name, std::mem::take(&mut points), None)
                .await?;
        }
        stdout.flush()?;
    }
    qdrant_client
        .upsert_points(&collection_name, points, None)
        .await?;
    Ok(())
}
