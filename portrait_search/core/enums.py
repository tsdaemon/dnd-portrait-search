from enum import StrEnum


class EmbedderType(StrEnum):
    INSTRUCTOR_LARGE_PATHFINDER_CHARACTER_INSTRUCTIONS = "instructor-large"
    ALL_MINI_LM_L6_V2 = "all-MiniLM-L6-v2"
    MS_MARCO_DISTILBERT_BASE_V4 = "msmarco-db-v4"


class SplitterType(StrEnum):
    LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_120_OVERLAP_60 = "langchain-recursive-c-120-o-60"


class DistanceType(StrEnum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dotProduct"
