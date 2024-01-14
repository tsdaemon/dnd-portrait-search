from collections import OrderedDict
from pathlib import Path
from typing import Any

from pydantic import Field, model_serializer, model_validator
from ruyaml import YAML

from portrait_search.core.entity import BaseEntity

yaml = YAML(typ="safe")
yaml.default_flow_style = False
yaml.indent(mapping=4, sequence=6, offset=4)


class PortraitMatch(BaseEntity):
    path: str = Field(min_length=1)
    match: set[str] = Field(min_length=1)

    @model_serializer
    def sort_model(self) -> dict[str, Any]:
        d2: dict[str, Any] = OrderedDict()
        d2["path"] = self.path
        d2["match"] = list(self.match)
        return d2


class Query(BaseEntity):
    query: str = Field(min_length=1)
    portraits: list[PortraitMatch] = Field(min_length=1)
    match: set[str] = Field(min_length=1)

    def add_portrait_match(self, path: str, match: set[str]) -> None:
        self.portraits.append(PortraitMatch(path=path, match=match))

    @model_validator(mode="after")
    def portraits_matches_subset_query_match(self) -> "Query":
        for portrait in self.portraits:
            if not portrait.match.issubset(self.match):
                raise ValueError(
                    f"Portrait {portrait.path} has matches {portrait.match} that are not in "
                    f"query {self.query} matches {self.match}"
                )
        return self

    @model_validator(mode="after")
    def should_have_portraits_with_same_path(self) -> "Query":
        existing_paths = set()
        for portrait in self.portraits:
            if portrait.path in existing_paths:
                raise ValueError(f"Portrait {portrait.path} is already in query {self.query}")
            existing_paths.add(portrait.path)
        return self

    @model_serializer
    def sort_model(self) -> dict[str, Any]:
        d2: dict[str, Any] = OrderedDict()
        d2["query"] = self.query
        d2["match"] = list(self.match)
        d2["portraits"] = self.portraits
        return d2


class DatasetEntry(BaseEntity):
    name: str = Field(min_length=1)
    queries: list[Query] = Field(min_length=1)

    def __repr__(self) -> str:
        return self.name

    @model_serializer
    def no_name(self) -> dict[str, Any]:
        return {"queries": self.queries}


def load_dataset(experiment: str) -> list[DatasetEntry]:
    path_to_dataset = Path(__file__).parent.parent.parent / "dataset" / experiment
    if not path_to_dataset.exists():
        raise ValueError(f"Dataset for {experiment} does not exist at path {path_to_dataset}")

    dataset_entries = []
    for yaml_file in path_to_dataset.glob("*.yaml"):
        with open(yaml_file) as file:
            yaml_data = yaml.load(file)
            dataset_entry = DatasetEntry(name=yaml_file.stem, **yaml_data)
            dataset_entries.append(dataset_entry)

    return dataset_entries


def store_dataset(dataset_entries: list[DatasetEntry], experiment: str) -> None:
    path_to_dataset = Path(__file__).parent.parent.parent / "dataset" / experiment
    path_to_dataset.mkdir(parents=True, exist_ok=True)

    for dataset_entry in dataset_entries:
        path_to_yaml = path_to_dataset / f"{dataset_entry.name}.yaml"
        with open(path_to_yaml, "w") as file:
            yaml.dump(dataset_entry.model_dump(), file)


def validate_dataset(dataset_entries: list[DatasetEntry], path_to_portraits: Path) -> None:
    for dataset_entry in dataset_entries:
        dataset_entry.validate_entity()
        _validate_dataset_entry(dataset_entry, path_to_portraits)


def _validate_dataset_entry(dataset_entry: DatasetEntry, path_to_portraits: Path) -> None:
    for query in dataset_entry.queries:
        for portrait in query.portraits:
            local_portrait_path = path_to_portraits / portrait.path
            if not local_portrait_path.exists():
                raise ValueError(f"Portrait {portrait.path} does not exist at path {local_portrait_path}")
