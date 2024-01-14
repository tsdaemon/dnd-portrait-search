from enum import StrEnum


class EmbedderType(StrEnum):
    INSTRUCTOR_LARGE_PATHFINDER_CHARACTER_INSTRUCTIONS = "instrtucor-large-pathfinder-character-instructions"
    ALL_MINI_LM_L6_V2 = "all-MiniLM-L6-v2"
    MS_MARCO_DISTILBERT_BASE_V4 = "msmarco-distilbert-base-v4"
    MS_MARCO_ROBERTA_BASE_ANCE_FIRSTP = "msmarco-roberta-base-ance-firstp"


class SplitterType(StrEnum):
    LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_100_OVERLAP_60 = "langchain-recursive-text-splitter-chunk-100-overlap-60"
    LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_120_OVERLAP_40 = "langchain-recursive-text-splitter-chunk-120-overlap-40"
    LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_120_OVERLAP_60 = "langchain-recursive-text-splitter-chunk-120-overlap-60"
    LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_160_OVERLAP_40 = "langchain-recursive-text-splitter-chunk-160-overlap-40"
    LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_160_OVERLAP_100 = "langchain-recursive-text-splitter-chunk-160-overlap-100"
    LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_200_OVERLAP_80 = "langchain-recursive-text-splitter-chunk-200-overlap-80"
    LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_300_OVERLAP_100 = "langchain-recursive-text-splitter-chunk-300-overlap-100"
    COMBINE_LCHUNK_160_O40_AND_LCHUNK_120_O60 = "combine-lchunk-160-o40-and-lchunk-120-o60"


class DistanceType(StrEnum):
    COSINE = "cosine"
    EUCLIDEAN = "eucledian"
    DOT_PRODUCT = "dot-product"
