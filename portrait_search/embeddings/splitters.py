import abc

from langchain.text_splitter import RecursiveCharacterTextSplitter


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


SPLITTERS: dict[str, type[TextSplitter]] = {
    "langchain-recursive-text-splitter-chunk-120-overlap-60": LangChainRecursiveTextSplitterChunk120Overlap60,
}
