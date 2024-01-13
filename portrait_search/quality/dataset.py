from pathlib import Path

import yaml
from pydantic import BaseModel


class PortraitMatch(BaseModel):
    path: str
    match: set[str]


class Query(BaseModel):
    query: str
    portraits: list[PortraitMatch]
    match: set[str]


class DatasetEntry(BaseModel):
    name: str
    queries: list[Query]

    def __repr__(self) -> str:
        return self.name


def load_dataset(experiment: str) -> list[DatasetEntry]:
    path_to_dataset = Path(__file__).parent.parent.parent / "dataset" / experiment
    if not path_to_dataset.exists():
        raise ValueError(f"Dataset for {experiment} does not exist at path {path_to_dataset}")

    dataset_entries = []
    for yaml_file in path_to_dataset.glob("*.yaml"):
        with open(yaml_file) as file:
            yaml_data = yaml.safe_load(file)
            dataset_entry = DatasetEntry(name=yaml_file.stem, **yaml_data)
            dataset_entries.append(dataset_entry)

    return dataset_entries


def validate_dataset(dataset_entries: list[DatasetEntry], path_to_portraits: Path) -> None:
    for dataset_entry in dataset_entries:
        _validate_dataset_entry(dataset_entry, path_to_portraits)


def _validate_dataset_entry(dataset_entry: DatasetEntry, path_to_portraits: Path) -> None:
    for query in dataset_entry.queries:
        if not query.match:
            raise ValueError(f"Query {query.query} has no matches, dataset entry {dataset_entry.name}")
        if not query.portraits:
            raise ValueError(f"Query {query.query} has no portraits, dataset entry {dataset_entry.name}")
        for portrait in query.portraits:
            if not portrait.match:
                raise ValueError(
                    f"Portrait {portrait.path} has no matches, query {query.query}, dataset entry {dataset_entry.name}"
                )
            if not portrait.match.issubset(query.match):
                local_portrait_path = path_to_portraits / portrait.path
                if not local_portrait_path.exists():
                    raise ValueError(f"Portrait {portrait.path} does not exist at path {local_portrait_path}")
                raise ValueError(
                    f"Portrait {portrait.path} has matches {portrait.match} that are not in "
                    f"query {query.query} matches {query.match}, dataset entry {dataset_entry.name}"
                )
