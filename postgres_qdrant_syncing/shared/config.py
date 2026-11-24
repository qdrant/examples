import os
from dataclasses import dataclass


@dataclass
class Config:
    # postgres
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "fashiondb")
    postgres_user: str = os.getenv("POSTGRES_USER", "postgres")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "postgres")

    # qdrant — defaults to local Docker instance
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "products")
    # Set CLOUD_INFERENCE=true to use Qdrant Cloud Inference (requires paid cluster).
    # When false (default), fastembed runs inference locally — no cloud account needed.
    cloud_inference: bool = os.getenv("CLOUD_INFERENCE", "false").lower() == "true"

    # embedding models
    dense_model: str = os.getenv("DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    sparse_model: str = os.getenv("SPARSE_MODEL", "qdrant/bm25")

    # tier 3 only
    kafka_bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    debezium_connect_url: str = os.getenv("DEBEZIUM_CONNECT_URL", "http://localhost:8083")

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def asyncpg_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


config = Config()