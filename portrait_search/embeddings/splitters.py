import abc

from langchain.text_splitter import RecursiveCharacterTextSplitter

from portrait_search.core.config import SplitterType


class TextSplitter(abc.ABC):
    @abc.abstractmethod
    def split(self, text: str) -> list[str]:
        raise NotImplementedError()


class LangChainRecursiveTextSplitter(TextSplitter, abc.ABC):
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def split(self, text: str) -> list[str]:
        return self.splitter.split_text(text)


class LangChainRecursiveTextSplitterChunk120Overlap60(LangChainRecursiveTextSplitter):
    def __init__(self) -> None:
        super().__init__(chunk_size=120, chunk_overlap=60)


SPLITTERS: dict[SplitterType, type[TextSplitter]] = {
    SplitterType.LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_120_OVERLAP_60: LangChainRecursiveTextSplitterChunk120Overlap60,  # noqa: E501
}
