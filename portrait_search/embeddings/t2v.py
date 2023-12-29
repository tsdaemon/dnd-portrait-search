from portrait_search.portraits.entities import PortraitRecord

from .embedders import EMBEDDERS
from .entities import EmbeddingRecord
from .splitters import SPLITTERS


def texts2vectors(
    portraits: list[PortraitRecord], splitter_name: str, embedder_name: str
) -> list[EmbeddingRecord]:
    splitter = SPLITTERS[splitter_name]()
    embedder = EMBEDDERS[embedder_name]()

    texts = [portrait.description for portrait in portraits]
    text_chunks, indices = zip(
        *[(chunk, i) for i, text in enumerate(texts) for chunk in splitter.split(text)]
    )
    embeddings = embedder.embed(text_chunks)  # type: ignore

    embeddings_records_by_portrait = {}
    for i, embedding, text_chunk in zip(indices, embeddings, text_chunks):
        portrait = portraits[i]
        if portrait.id not in embeddings_records_by_portrait:
            embeddings_records_by_portrait[portrait.id] = EmbeddingRecord(
                portrait_id=portrait.id,
                embeddings=[],
                embedded_texts=[],
                splitter_class=splitter_name,
                embedding_model_class=embedder_name,
            )
        embeddings_records_by_portrait[portrait.id].embeddings.append(embedding)
        embeddings_records_by_portrait[portrait.id].embedded_texts.append(text_chunk)

    return list(embeddings_records_by_portrait.values())
