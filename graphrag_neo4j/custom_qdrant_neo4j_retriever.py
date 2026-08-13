from __future__ import annotations

import logging
from typing import Any, Optional

import neo4j
from neo4j_graphrag.exceptions import EmbeddingRequiredError, SearchValidationError
from neo4j_graphrag.retrievers import QdrantNeo4jRetriever
from neo4j_graphrag.retrievers.external.utils import get_match_query
from neo4j_graphrag.types import RawSearchResult, VectorSearchModel
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class CustomQdrantNeo4jRetriever(QdrantNeo4jRetriever):
    """
    Custom retriever inheriting from QdrantNeo4jRetriever.
    Handles cases where the external ID in Qdrant payload might be a list.

    Inherits initialization and other methods from QdrantNeo4jRetriever.
    Only overrides the get_search_results method for custom logic.
    """

    def get_search_results(
        self,
        query_vector: Optional[list[float]] = None,
        query_text: Optional[str] = None,
        top_k: int = 5,
        **kwargs: Any,
    ) -> RawSearchResult:
        try:
            validated_data = VectorSearchModel(
                query_vector=query_vector,
                query_text=query_text,
                top_k=top_k,
            )
        except ValidationError as e:
            raise SearchValidationError(e.errors()) from e

        if validated_data.query_text:
            if self.embedder:
                query_vector = self.embedder.embed_query(validated_data.query_text)
                logger.debug("Locally generated query vector: %s", query_vector)
            else:
                logger.error("No embedder provided for query_text.")
                raise EmbeddingRequiredError("No embedder provided for query_text.")

        points = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=[self.id_property_external],
            **kwargs,
        ).points

        # Custom logic
        result_tuples = []
        for point in points:
            assert point.payload is not None
            target_ids = point.payload.get(self.id_property_external, [point.id])
            result_tuples = [[target_id, point.score] for target_id in target_ids]

        search_query = get_match_query(
            return_properties=self.return_properties,
            retrieval_query=self.retrieval_query,
        )

        parameters = {
            "match_params": result_tuples,
            "id_property": self.id_property_neo4j,
        }

        logger.debug("Qdrant Store Cypher parameters: %s", parameters)
        logger.debug("Qdrant Store Cypher query: %s", search_query)

        records, _, _ = self.driver.execute_query(
            search_query,
            parameters,
            database_=self.neo4j_database,
            routing_=neo4j.RoutingControl.READ,
        )

        return RawSearchResult(records=records)
