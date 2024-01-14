import abc

from langchain.text_splitter import RecursiveCharacterTextSplitter

from portrait_search.core.enums import SplitterType


class Splitter(abc.ABC):
    def __init__(self) -> None:
        self._type: SplitterType | None = None

    def __repr__(self) -> str:
        return f"Splitter {self._type}"

    @property
    def type(self) -> SplitterType:
        if self._type is None:
            raise ValueError("Splitter type not set")
        return self._type

    @type.setter
    def type(self, value: SplitterType) -> None:
        self._type = value

    @abc.abstractmethod
    def split(self, text: str) -> list[str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def split_query(self, text: str) -> list[str]:
        raise NotImplementedError()


SPLITTERS: dict[SplitterType, Splitter] = {}


def register_splitter(splitter_type: SplitterType, splitter: Splitter) -> None:
    SPLITTERS[splitter_type] = splitter
    splitter.type = splitter_type


class DoNotSplitQueryMixin(Splitter, abc.ABC):
    def split_query(self, text: str) -> list[str]:
        return [text]


class CombineSplitter(Splitter):
    def __init__(self, splitters: list[Splitter]) -> None:
        self.splitters = splitters

    def split(self, text: str) -> list[str]:
        return [chunk for splitter in self.splitters for chunk in splitter.split(text)]

    def split_query(self, text: str) -> list[str]:
        return [chunk for splitter in self.splitters for chunk in splitter.split_query(text)]


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


# register_splitter(
#     SplitterType.LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_300_OVERLAP_100,
#     LangChainRecursiveSplitter(chunk_size=300, chunk_overlap=100),
# )
# register_splitter(
#     SplitterType.LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_200_OVERLAP_80,
#     LangChainRecursiveSplitter(chunk_size=200, chunk_overlap=80),
# )
register_splitter(
    SplitterType.LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_160_OVERLAP_40,
    LangChainRecursiveSplitter(chunk_size=160, chunk_overlap=40),
)
register_splitter(
    SplitterType.LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_120_OVERLAP_60,
    LangChainRecursiveSplitter(chunk_size=120, chunk_overlap=60),
)
register_splitter(
    SplitterType.LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_100_OVERLAP_60,
    LangChainRecursiveSplitter(chunk_size=100, chunk_overlap=60),
)
# register_splitter(
#     SplitterType.LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_120_OVERLAP_40,
#     LangChainRecursiveSplitter(chunk_size=120, chunk_overlap=40),
# )
# register_splitter(
#     SplitterType.COMBINE_LCHUNK_160_O40_AND_LCHUNK_120_O60,
#     CombineSplitter(
#         [
#             LangChainRecursiveSplitter(chunk_size=160, chunk_overlap=40),
#             LangChainRecursiveSplitter(chunk_size=120, chunk_overlap=60),
#         ]
#     ),
# )
