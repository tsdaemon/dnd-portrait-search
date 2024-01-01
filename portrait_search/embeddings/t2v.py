from portrait_search.core.config import EmbedderType, SplitterType
from portrait_search.portraits.entities import PortraitRecord

from .embedders import Embedder
from .entities import EmbeddingRecord
from .splitters import TextSplitter


def portraits2embeddings(
    portraits: list[PortraitRecord],
    splitter: TextSplitter,
    splitter_type: SplitterType,
    embedder: Embedder,
    embedder_type: EmbedderType,
) -> list[EmbeddingRecord]:
    if not portraits:
        return []

    texts = [portrait.description for portrait in portraits]
    text_chunks, indices = zip(*[(chunk, i) for i, text in enumerate(texts) for chunk in splitter.split(text)])
    embeddings = embedder.embed(text_chunks)  # type: ignore

    embeddings_records = []
    for i, embedding, text_chunk in zip(indices, embeddings, text_chunks):
        portrait = portraits[i]
        embedding_record = EmbeddingRecord(
            portrait_id=portrait.id,
            embedding=embedding,
            embedded_text=text_chunk,
            splitter_type=splitter_type,
            embedder_type=embedder_type,
        )
        embeddings_records.append(embedding_record)

    return embeddings_records
