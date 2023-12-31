from collections import defaultdict

import numpy as np

from portrait_search.core.enums import DistanceType
from portrait_search.embeddings.embedders import Embedder
from portrait_search.embeddings.entities import EmbeddingSimilarity
from portrait_search.embeddings.repository import EmbeddingRepository
from portrait_search.embeddings.splitters import Splitter
from portrait_search.embeddings.t2v import query2embeddings
from portrait_search.portraits.entities import PortraitRecord
from portrait_search.portraits.repository import PortraitRepository

from .retriever import Retriever


class SimilarityRetriever(Retriever):
    def __init__(
        self,
        distance_type: DistanceType,
        embedding_repository: EmbeddingRepository,
        portrait_repository: PortraitRepository,
        splitter: Splitter,
        embedder: Embedder,
    ) -> None:
        self.distance_type = distance_type
        self.embedding_repository = embedding_repository
        self.portrait_repository = portrait_repository
        self.splitter = splitter
        self.embedder = embedder

    async def get_portraits(
        self, query: str, experiment: str | None = None, limit: int = 10
    ) -> tuple[list[PortraitRecord], list[list[EmbeddingSimilarity]]]:
        """Returns a list of PortraitRecords that match the query string."""

        # First get the embeddings for the query
        qeury_embeddings_and_texts = query2embeddings(query, self.splitter, self.embedder)

        # Then search for the embeddings in the database
        all_embedding_similarities = []
        for query_embedding_and_text in zip(*qeury_embeddings_and_texts):
            query_embedding, query_text = query_embedding_and_text
            embedding_similarities = await self.embedding_repository.vector_search(
                query_embedding,
                self.splitter.splitter_type(),
                self.embedder.embedder_type(),
                self.distance_type,
                experiment=experiment,
                # Get more candidates to widen search
                limit=limit * 3,
            )
            for embedding_similarity in embedding_similarities:
                embedding_similarity.query_text = query_text
                embedding_similarity.query = query_embedding

            all_embedding_similarities.extend(embedding_similarities)

        # Find all similarities for all unique portraits
        portrait_similarity = defaultdict(list)
        portrait_similarity_explanation = defaultdict(list)
        for embedding_similarity in all_embedding_similarities:
            portrait_similarity[embedding_similarity.portrait_id].append(embedding_similarity.similarity)
            portrait_similarity_explanation[embedding_similarity.portrait_id].append(embedding_similarity)

        # Calculate the average similarity for each portrait
        portrait_similarity_average = {k: np.mean(v) for k, v in portrait_similarity.items()}

        # Take top N with the highest similarity where N is the limit
        top_portrait_ids = [
            p for p, _ in sorted(portrait_similarity_average.items(), key=lambda item: item[1], reverse=True)
        ][:limit]

        portraits = [await self.portrait_repository.get_one(portrait_id) for portrait_id in top_portrait_ids]
        similarity_explanations = [portrait_similarity_explanation[portrait_id] for portrait_id in top_portrait_ids]
        return portraits, similarity_explanations
