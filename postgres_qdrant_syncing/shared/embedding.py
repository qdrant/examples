import uuid


def build_embedding_text(product: dict) -> str:
    """
    Combine product fields into a single string for embedding.
    """
    parts = [
        product.get("name", ""),
        product.get("description", ""),
        product.get("product_type", ""),
        product.get("color", ""),
        product.get("department", ""),
    ]
    return " | ".join(p for p in parts if p)


def article_id_to_uuid(article_id: str) -> str:
    """
    Convert an article_id string to a deterministic UUID for use as a Qdrant point ID.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_OID, article_id))