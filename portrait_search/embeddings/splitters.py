import abc

from langchain.text_splitter import RecursiveCharacterTextSplitter

from portrait_search.core.enums import SplitterType


class Splitter(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def splitter_type(cls) -> SplitterType:
        raise NotImplementedError()

    @abc.abstractmethod
    def split(self, text: str) -> list[str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def split_query(self, text: str) -> list[str]:
        raise NotImplementedError()


class DoNotSplitQueryMixin(Splitter, abc.ABC):
    def split_query(self, text: str) -> list[str]:
        return [text]


class LangChainRecursiveSplitter(Splitter):
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def split(self, text: str) -> list[str]:
        return self.splitter.split_text(text)

    def split_query(self, text: str) -> list[str]:
        return self.splitter.split_text(text)


class LangChainRecursiveTextSplitterChunk120Overlap40(LangChainRecursiveSplitter):
    def __init__(self) -> None:
        super().__init__(chunk_size=120, chunk_overlap=40)

    @classmethod
    def splitter_type(cls) -> SplitterType:
        return SplitterType.LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_120_OVERLAP_40


class LangChainRecursiveTextSplitterChunk120Overlap60(LangChainRecursiveSplitter):
    def __init__(self) -> None:
        super().__init__(chunk_size=120, chunk_overlap=60)

    @classmethod
    def splitter_type(cls) -> SplitterType:
        return SplitterType.LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_120_OVERLAP_60


class LangChainRecursiveTextSplitterChunk160Overlap40(LangChainRecursiveSplitter):
    def __init__(self) -> None:
        super().__init__(chunk_size=120, chunk_overlap=60)

    @classmethod
    def splitter_type(cls) -> SplitterType:
        return SplitterType.LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_160_OVERLAP_40

    def split_query(self, text: str) -> list[str]:
        return self.splitter.split_text(text)


class LangChainRecursiveTextSplitterChunk200Overlap80(LangChainRecursiveSplitter):
    def __init__(self) -> None:
        super().__init__(chunk_size=200, chunk_overlap=80)

    @classmethod
    def splitter_type(cls) -> SplitterType:
        return SplitterType.LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_200_OVERLAP_80


class LangChainRecursiveTextSplitterChunk300Overlap100(LangChainRecursiveSplitter):
    def __init__(self) -> None:
        super().__init__(chunk_size=300, chunk_overlap=100)

    @classmethod
    def splitter_type(cls) -> SplitterType:
        return SplitterType.LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_300_OVERLAP_100


SPLITTERS: dict[SplitterType, type[Splitter]] = {
    t.splitter_type(): t  # type: ignore[type-abstract]
    for t in [
        LangChainRecursiveTextSplitterChunk120Overlap40,
        LangChainRecursiveTextSplitterChunk120Overlap60,
        LangChainRecursiveTextSplitterChunk160Overlap40,
        LangChainRecursiveTextSplitterChunk200Overlap80,
        LangChainRecursiveTextSplitterChunk300Overlap100,
    ]
}
